// Package tensor implements a minimal N-dimensional float32 tensor:
// shape + strides over a flat buffer, zero-copy views (Reshape, Transpose),
// broadcasting element-wise ops and matrix multiplication.
//
// This is the numeric foundation for the whole course (lecture L1).
package tensor

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
)

// Tensor is a view over a flat float32 buffer.
// Element (i0, i1, ..., in) lives at data[offset + i0*strides[0] + ... + in*strides[n]].
type Tensor struct {
	data    []float32
	shape   []int
	strides []int
	offset  int
}

// New returns a zero-filled contiguous tensor of the given shape.
func New(shape ...int) *Tensor {
	return FromSlice(make([]float32, numel(shape)), shape...)
}

// FromSlice wraps data (without copying) as a contiguous tensor of the given shape.
func FromSlice(data []float32, shape ...int) *Tensor {
	if len(data) != numel(shape) {
		panic(fmt.Sprintf("tensor: FromSlice: %d values do not fit shape %v", len(data), shape))
	}
	return &Tensor{data: data, shape: clone(shape), strides: rowMajorStrides(shape)}
}

// Randn returns a tensor with elements drawn from N(0, 1).
func Randn(rng *rand.Rand, shape ...int) *Tensor {
	t := New(shape...)
	for i := range t.data {
		t.data[i] = float32(rng.NormFloat64())
	}
	return t
}

func (t *Tensor) Shape() []int   { return clone(t.shape) }
func (t *Tensor) Strides() []int { return clone(t.strides) }
func (t *Tensor) Dims() int      { return len(t.shape) }
func (t *Tensor) Size() int      { return numel(t.shape) }

// At returns the element at the given multi-index.
func (t *Tensor) At(idx ...int) float32 { return t.data[t.index(idx)] }

// Set writes v at the given multi-index. Writes through views are visible in the parent.
func (t *Tensor) Set(v float32, idx ...int) { t.data[t.index(idx)] = v }

func (t *Tensor) index(idx []int) int {
	if len(idx) != len(t.shape) {
		panic(fmt.Sprintf("tensor: index %v for shape %v", idx, t.shape))
	}
	off := t.offset
	for d, i := range idx {
		if i < 0 || i >= t.shape[d] {
			panic(fmt.Sprintf("tensor: index %v out of range for shape %v", idx, t.shape))
		}
		off += i * t.strides[d]
	}
	return off
}

// IsContiguous reports whether elements are laid out densely in row-major order.
func (t *Tensor) IsContiguous() bool {
	expected := 1
	for d := len(t.shape) - 1; d >= 0; d-- {
		if t.shape[d] != 1 && t.strides[d] != expected {
			return false
		}
		expected *= t.shape[d]
	}
	return true
}

// Contiguous returns t itself if it is already contiguous, otherwise a dense copy.
func (t *Tensor) Contiguous() *Tensor {
	if t.IsContiguous() {
		return t
	}
	out := New(t.shape...)
	i := 0
	t.each(func(off int) {
		out.data[i] = t.data[off]
		i++
	})
	return out
}

// Data returns the elements in row-major order. For a contiguous tensor the
// returned slice aliases the underlying buffer.
func (t *Tensor) Data() []float32 {
	c := t.Contiguous()
	return c.data[c.offset : c.offset+c.Size()]
}

// Reshape returns a view with a new shape. One dimension may be -1 and is inferred.
// Non-contiguous tensors are copied first (as torch.reshape does).
func (t *Tensor) Reshape(shape ...int) *Tensor {
	shape = clone(shape)
	infer, known := -1, 1
	for d, s := range shape {
		if s == -1 {
			if infer >= 0 {
				panic("tensor: Reshape: more than one -1")
			}
			infer = d
			continue
		}
		known *= s
	}
	if infer >= 0 {
		shape[infer] = t.Size() / known
	}
	if numel(shape) != t.Size() {
		panic(fmt.Sprintf("tensor: Reshape %v -> %v changes size", t.shape, shape))
	}
	c := t.Contiguous()
	return &Tensor{data: c.data, shape: shape, strides: rowMajorStrides(shape), offset: c.offset}
}

// Transpose swaps dimensions a and b. No data is copied: only shape and strides are swapped.
func (t *Tensor) Transpose(a, b int) *Tensor {
	v := &Tensor{data: t.data, shape: clone(t.shape), strides: clone(t.strides), offset: t.offset}
	v.shape[a], v.shape[b] = v.shape[b], v.shape[a]
	v.strides[a], v.strides[b] = v.strides[b], v.strides[a]
	return v
}

// T transposes the last two dimensions (matrix transpose, batched if Dims() > 2).
func (t *Tensor) T() *Tensor {
	n := len(t.shape)
	return t.Transpose(n-2, n-1)
}

// each calls f with the flat offset of every element, in row-major order of t's shape.
func (t *Tensor) each(f func(off int)) {
	eachIndex(t.shape, func(idx []int) {
		off := t.offset
		for d, i := range idx {
			off += i * t.strides[d]
		}
		f(off)
	})
}

// eachIndex iterates over all multi-indices of shape in row-major order
// (like an odometer: the last digit spins fastest).
func eachIndex(shape []int, f func(idx []int)) {
	if numel(shape) == 0 {
		return
	}
	idx := make([]int, len(shape))
	for {
		f(idx)
		d := len(shape) - 1
		for ; d >= 0; d-- {
			idx[d]++
			if idx[d] < shape[d] {
				break
			}
			idx[d] = 0
		}
		if d < 0 {
			return
		}
	}
}

// AllClose reports whether a and b have equal shapes and |a-b| <= atol + rtol*|b| element-wise.
func AllClose(a, b *Tensor, rtol, atol float64) bool {
	if !equalInts(a.shape, b.shape) {
		return false
	}
	ad, bd := a.Data(), b.Data()
	for i := range ad {
		x, y := float64(ad[i]), float64(bd[i])
		if math.Abs(x-y) > atol+rtol*math.Abs(y) {
			return false
		}
	}
	return true
}

func (t *Tensor) String() string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "Tensor%v ", t.shape)
	if t.Size() > 64 {
		sb.WriteString("[...]")
		return sb.String()
	}
	fmt.Fprint(&sb, t.Data())
	return sb.String()
}

func rowMajorStrides(shape []int) []int {
	s := make([]int, len(shape))
	acc := 1
	for d := len(shape) - 1; d >= 0; d-- {
		s[d] = acc
		acc *= shape[d]
	}
	return s
}

func numel(shape []int) int {
	n := 1
	for _, s := range shape {
		n *= s
	}
	return n
}

func clone(s []int) []int { return append([]int(nil), s...) }

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
