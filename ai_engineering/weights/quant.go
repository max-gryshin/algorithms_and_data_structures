package weights

import (
	"math"
	"runtime"
	"sync"
)

// Quantization (L12). Matrices are stored "one row per output channel": (out, in),
// the layout of HF/PyTorch and of our tied embedding. y = W·x is then one dot
// product per output, reading the weights strictly sequentially.

// QParams map reals to integers: q = clamp(round(x/Scale) + Zero, Min, Max), x ≈ (q − Zero)·Scale.
type QParams struct {
	Scale    float32
	Zero     int32
	Min, Max int32
}

// Symmetric: zero point 0, range [−(2^(b−1)−1), 2^(b−1)−1], Scale = absmax / qmax.
func Symmetric(absmax float32, bits int) QParams {
	qmax := int32(1)<<(bits-1) - 1
	s := absmax / float32(qmax)
	if s == 0 {
		s = 1
	}
	return QParams{Scale: s, Min: -qmax, Max: qmax}
}

// Asymmetric: range [0, 2^b − 1] covers [lo, hi]; the zero point makes 0.0 exact.
func Asymmetric(lo, hi float32, bits int) QParams {
	lo, hi = min(lo, 0), max(hi, 0)
	qmax := int32(1)<<bits - 1
	s := (hi - lo) / float32(qmax)
	if s == 0 {
		s = 1
	}
	z := int32(math.Round(float64(-lo / s)))
	return QParams{Scale: s, Zero: z, Min: 0, Max: qmax}
}

func (p QParams) Quantize(x float32) int32 {
	q := int32(math.Round(float64(x/p.Scale))) + p.Zero
	return max(p.Min, min(p.Max, q))
}

func (p QParams) Dequantize(q int32) float32 { return float32(q-p.Zero) * p.Scale }

// Matrix is a weight matrix (out, in) in some storage format.
type Matrix interface {
	Dims() (out, in int)
	MulVec(dst, x []float32) // dst[o] = Σ_i W[o,i]·x[i]
	Dequantize() []float32   // (out, in) float32
	Bytes() int              // storage size
}

// F32Matrix is the unquantized baseline.
type F32Matrix struct {
	Out, In int
	W       []float32
}

func (m *F32Matrix) Dims() (int, int)      { return m.Out, m.In }
func (m *F32Matrix) Dequantize() []float32 { return m.W }
func (m *F32Matrix) Bytes() int            { return 4 * len(m.W) }
func (m *F32Matrix) MulVec(dst, x []float32) {
	parallelRows(m.Out, m.In, func(o int) {
		row := m.W[o*m.In : (o+1)*m.In]
		var s0, s1, s2, s3 float32
		i := 0
		for ; i+4 <= len(row); i += 4 {
			s0 += row[i] * x[i]
			s1 += row[i+1] * x[i+1]
			s2 += row[i+2] * x[i+2]
			s3 += row[i+3] * x[i+3]
		}
		for ; i < len(row); i++ {
			s0 += row[i] * x[i]
		}
		dst[o] = s0 + s1 + s2 + s3
	})
}

// Int8Matrix: symmetric per-channel int8, one float32 scale per output row.
type Int8Matrix struct {
	Out, In int
	Q       []int8
	Scale   []float32
}

// QuantizeInt8 quantizes w (out, in) per row; perTensor uses one scale for all rows.
func QuantizeInt8(w []float32, out, in int, perTensor bool) *Int8Matrix {
	m := &Int8Matrix{Out: out, In: in, Q: make([]int8, len(w)), Scale: make([]float32, out)}
	var global float32
	for _, v := range w {
		global = max(global, abs(v))
	}
	for o := range out {
		row := w[o*in : (o+1)*in]
		amax := global
		if !perTensor {
			amax = 0
			for _, v := range row {
				amax = max(amax, abs(v))
			}
		}
		p := Symmetric(amax, 8)
		m.Scale[o] = p.Scale
		for i, v := range row {
			m.Q[o*in+i] = int8(p.Quantize(v))
		}
	}
	return m
}

func (m *Int8Matrix) Dims() (int, int) { return m.Out, m.In }
func (m *Int8Matrix) Bytes() int       { return len(m.Q) + 4*len(m.Scale) }
func (m *Int8Matrix) Dequantize() []float32 {
	w := make([]float32, len(m.Q))
	for i, q := range m.Q {
		w[i] = float32(q) * m.Scale[i/m.In]
	}
	return w
}
func (m *Int8Matrix) MulVec(dst, x []float32) {
	parallelRows(m.Out, m.In, func(o int) {
		row := m.Q[o*m.In : (o+1)*m.In]
		var s0, s1, s2, s3 float32
		i := 0
		for ; i+4 <= len(row); i += 4 {
			s0 += float32(row[i]) * x[i]
			s1 += float32(row[i+1]) * x[i+1]
			s2 += float32(row[i+2]) * x[i+2]
			s3 += float32(row[i+3]) * x[i+3]
		}
		for ; i < len(row); i++ {
			s0 += float32(row[i]) * x[i]
		}
		dst[o] = (s0 + s1 + s2 + s3) * m.Scale[o] // the scale is factored out of the sum
	})
}

// Int4Matrix: Q4_0-like blocks of Group values along a row. Per block one fp16 scale
// d = m/−8 (m is the value with the largest |·|) and Group 4-bit codes q ∈ [0, 15],
// x ≈ (q − 8)·d; two codes per byte (low nibble first). Group 32 → 4.5 bits/weight.
type Int4Matrix struct {
	Out, In, Group int
	Q              []byte   // out·in/2
	D              []uint16 // fp16 scales, out·in/Group
}

// QuantizeInt4 needs in % group == 0 and an even group.
func QuantizeInt4(w []float32, out, in, group int) *Int4Matrix {
	if in%group != 0 || group%2 != 0 {
		panic("weights: QuantizeInt4: in must be a multiple of an even group")
	}
	m := &Int4Matrix{Out: out, In: in, Group: group, Q: make([]byte, len(w)/2), D: make([]uint16, len(w)/group)}
	for blk := range m.D {
		x := w[blk*group : (blk+1)*group]
		var mx float32
		for _, v := range x {
			if abs(v) > abs(mx) {
				mx = v
			}
		}
		m.D[blk] = F32ToF16(mx / -8)
		d := F16ToF32(m.D[blk]) // quantize with the scale the kernel will see
		var inv float32
		if d != 0 {
			inv = 1 / d
		}
		for j := 0; j < group; j += 2 {
			q0 := min(15, max(0, int(math.Round(float64(x[j]*inv)))+8))
			q1 := min(15, max(0, int(math.Round(float64(x[j+1]*inv)))+8))
			m.Q[(blk*group+j)/2] = byte(q0) | byte(q1)<<4
		}
	}
	return m
}

func (m *Int4Matrix) Dims() (int, int) { return m.Out, m.In }
func (m *Int4Matrix) Bytes() int       { return len(m.Q) + 2*len(m.D) }
func (m *Int4Matrix) Dequantize() []float32 {
	w := make([]float32, len(m.Q)*2)
	for i, b := range m.Q {
		d := F16ToF32(m.D[2*i/m.Group])
		w[2*i] = float32(int(b&15)-8) * d
		w[2*i+1] = float32(int(b>>4)-8) * d
	}
	return w
}
func (m *Int4Matrix) MulVec(dst, x []float32) {
	nb := m.In / m.Group
	parallelRows(m.Out, m.In, func(o int) {
		var s float32
		for g := range nb {
			blk := o*nb + g
			q := m.Q[blk*m.Group/2 : (blk+1)*m.Group/2]
			xs := x[g*m.Group : (g+1)*m.Group]
			var a0, a1 float32
			for j, b := range q {
				a0 += float32(int(b&15)-8) * xs[2*j]
				a1 += float32(int(b>>4)-8) * xs[2*j+1]
			}
			s += (a0 + a1) * F16ToF32(m.D[blk]) // one multiply by d per block
		}
		dst[o] = s
	})
}

// parallelRows splits output rows between goroutines for large matrices.
func parallelRows(out, in int, f func(o int)) {
	workers := runtime.GOMAXPROCS(0)
	if out*in < 1<<16 || workers == 1 {
		for o := range out {
			f(o)
		}
		return
	}
	chunk := (out + workers - 1) / workers
	var wg sync.WaitGroup
	for from := 0; from < out; from += chunk {
		to := min(from+chunk, out)
		wg.Go(func() {
			for o := from; o < to; o++ {
				f(o)
			}
		})
	}
	wg.Wait()
}

func abs(x float32) float32 { return math.Float32frombits(math.Float32bits(x) &^ (1 << 31)) }

// Format selects a storage format for QuantizeModel.
type Format string

const (
	FormatF32  Format = "f32"
	FormatInt8 Format = "int8"
	FormatInt4 Format = "int4"
)

// Quantize stores w (out, in) in the given format.
func Quantize(w []float32, out, in int, f Format) Matrix {
	switch f {
	case FormatInt8:
		return QuantizeInt8(w, out, in, false)
	case FormatInt4:
		return QuantizeInt4(w, out, in, 32)
	}
	return &F32Matrix{Out: out, In: in, W: w}
}
