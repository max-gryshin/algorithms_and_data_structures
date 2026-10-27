package tensor

import "fmt"

// BroadcastShapes returns the shape produced by broadcasting a and b (NumPy rules):
// shapes are aligned from the right; two dimensions are compatible when equal or one of them is 1.
func BroadcastShapes(a, b []int) ([]int, error) {
	n := max(len(a), len(b))
	out := make([]int, n)
	for i := 1; i <= n; i++ {
		da, db := dimFromRight(a, i), dimFromRight(b, i)
		switch {
		case da == db, db == 1:
			out[n-i] = da
		case da == 1:
			out[n-i] = db
		default:
			return nil, fmt.Errorf("tensor: cannot broadcast %v with %v", a, b)
		}
	}
	return out, nil
}

func dimFromRight(shape []int, i int) int {
	if i > len(shape) {
		return 1
	}
	return shape[len(shape)-i]
}

// broadcastStrides returns strides that let t be read as if it had shape out:
// a broadcast dimension gets stride 0, so the same element is reused.
func (t *Tensor) broadcastStrides(out []int) []int {
	s := make([]int, len(out))
	shift := len(out) - len(t.shape)
	for d := range t.shape {
		if t.shape[d] != 1 {
			s[d+shift] = t.strides[d]
		}
	}
	return s
}

// zipWith applies f element-wise to a and b with broadcasting.
func zipWith(a, b *Tensor, f func(x, y float32) float32) *Tensor {
	shape, err := BroadcastShapes(a.shape, b.shape)
	if err != nil {
		panic(err)
	}
	sa, sb := a.broadcastStrides(shape), b.broadcastStrides(shape)
	out := New(shape...)
	i := 0
	eachIndex(shape, func(idx []int) {
		oa, ob := a.offset, b.offset
		for d, k := range idx {
			oa += k * sa[d]
			ob += k * sb[d]
		}
		out.data[i] = f(a.data[oa], b.data[ob])
		i++
	})
	return out
}

func Add(a, b *Tensor) *Tensor { return zipWith(a, b, func(x, y float32) float32 { return x + y }) }
func Sub(a, b *Tensor) *Tensor { return zipWith(a, b, func(x, y float32) float32 { return x - y }) }
func Mul(a, b *Tensor) *Tensor { return zipWith(a, b, func(x, y float32) float32 { return x * y }) }

// Map returns a new tensor with f applied to every element.
func (t *Tensor) Map(f func(float32) float32) *Tensor {
	out := New(t.shape...)
	i := 0
	t.each(func(off int) {
		out.data[i] = f(t.data[off])
		i++
	})
	return out
}

func (t *Tensor) Scale(s float32) *Tensor { return t.Map(func(x float32) float32 { return x * s }) }

// Sum returns the sum of all elements (accumulated in float64 for accuracy).
func (t *Tensor) Sum() float64 {
	var s float64
	t.each(func(off int) { s += float64(t.data[off]) })
	return s
}

// Dot returns the scalar product of two 1-D tensors of equal length.
func Dot(a, b *Tensor) float32 {
	if a.Dims() != 1 || b.Dims() != 1 || a.shape[0] != b.shape[0] {
		panic(fmt.Sprintf("tensor: Dot needs two equal 1-D tensors, got %v and %v", a.shape, b.shape))
	}
	var s float32
	for i := range a.shape[0] {
		s += a.data[a.offset+i*a.strides[0]] * b.data[b.offset+i*b.strides[0]]
	}
	return s
}
