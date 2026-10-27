package autograd

import (
	"runtime"
	"slices"
	"sync"
	"sync/atomic"

	"algorithms_and_data_structures/ai_engineering/tensor"
)

// Fast paths for training GoGPT on a CPU (lecture L9). Two things made the
// straightforward ops slow on small transformers:
//
//  1. (B, T, C)·(C, D) went through the batched tensor.MatMul: B separate T×C·C×D
//     products, each too small to be split between cores, and a goroutine fan-out per
//     product that cost more than the arithmetic. Flattening to one (B·T, C)·(C, D)
//     product and splitting its rows between cores fixes both.
//  2. Every broadcast (bias add, mask add) materialized a full-size copy first.
//
// None of this changes the arithmetic: each output element is still accumulated in
// the same order, so results do not depend on the number of threads.

var threads atomic.Int64

// SetThreads limits how many goroutines one op may use (intra-op parallelism) and
// returns the previous limit. n ≤ 0 restores the default, GOMAXPROCS. Data-parallel
// training (L9) sets it to 1 to measure the speedup of the workers alone.
func SetThreads(n int) int {
	old := Threads()
	if n <= 0 {
		n = 0
	}
	threads.Store(int64(n))
	return old
}

// Threads is the current intra-op limit.
func Threads() int {
	if n := int(threads.Load()); n > 0 {
		return n
	}
	return runtime.GOMAXPROCS(0)
}

// minWorkPerThread is the number of multiply-adds below which a goroutine costs
// more than it saves (~10 µs of work).
const minWorkPerThread = 32 * 1024

// parallelFor splits [0, n) into contiguous chunks and runs f on them concurrently.
// work is the total cost (multiply-adds); small jobs run on the calling goroutine.
func parallelFor(n, work int, f func(lo, hi int)) {
	w := min(Threads(), n, max(1, work/minWorkPerThread))
	if w <= 1 {
		f(0, n)
		return
	}
	chunk := (n + w - 1) / w
	var wg sync.WaitGroup
	for lo := chunk; lo < n; lo += chunk {
		wg.Go(func() { f(lo, min(lo+chunk, n)) })
	}
	f(0, min(chunk, n))
	wg.Wait()
}

// mm computes out = a·b for row-major a (n×k), b (k×p); out (n×p) is zeroed here.
// Rows of out are split between threads; every row uses the tiled kernel of L1.
func mm(a, b []float32, n, k, p int) []float32 {
	out := make([]float32, n*p)
	parallelFor(n, n*k*p, func(lo, hi int) {
		tensor.MatMulTiled(a[lo*k:hi*k], b, out[lo*p:hi*p], hi-lo, k, p)
	})
	return out
}

// mmBT computes out = a·bᵀ for a (n×k), b (p×k): out[i][j] = a_i · b_j.
// Both operands are read row by row, so no transposed copy is needed.
func mmBT(a, b []float32, n, k, p int) []float32 {
	out := make([]float32, n*p)
	parallelFor(n, n*k*p, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			ai := a[i*k : (i+1)*k]
			orow := out[i*p : (i+1)*p]
			for j := range p {
				orow[j] = dot(ai, b[j*k:(j+1)*k])
			}
		}
	})
	return out
}

// mmAT computes out = aᵀ·b for a (n×k), b (n×p): out (k×p) = Σ_i a_iᵀ b_i.
// a is transposed once (O(n·k)), then the tiled kernel runs on (k×n)·(n×p).
func mmAT(a, b []float32, n, k, p int) []float32 {
	return mm(transpose2D(a, n, k), b, k, n, p)
}

// dot is a·b with four independent accumulators: a single running sum is one long
// chain of dependent multiply-adds, four chains keep the CPU's FMA units busy.
func dot(a, b []float32) float32 {
	b = b[:len(a)]
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= len(a); i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for ; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return (s0 + s1) + (s2 + s3)
}

func transpose2D(a []float32, n, k int) []float32 {
	out := make([]float32, n*k)
	for i := range n {
		for j, x := range a[i*k : (i+1)*k] {
			out[j*n+i] = x
		}
	}
	return out
}

// swapAxes copies data of shape (pre, A, B, post) into (pre, B, A, post) — the
// transpose of two adjacent axes, moving `post` contiguous floats at a time.
func swapAxes(src []float32, pre, a, b, post int) []float32 {
	dst := make([]float32, len(src))
	for p := range pre {
		base := p * a * b * post
		for i := range a {
			for j := range b {
				s := base + (i*b+j)*post
				d := base + (j*a+i)*post
				copy(dst[d:d+post], src[s:s+post])
			}
		}
	}
	return dst
}

// transposeData returns t with axes d1 and d2 swapped, as a contiguous tensor.
func transposeData(t *tensor.Tensor, d1, d2 int) *tensor.Tensor {
	if d1 > d2 {
		d1, d2 = d2, d1
	}
	shape := t.Shape()
	if d2 != d1+1 {
		return t.Transpose(d1, d2).Contiguous()
	}
	pre, post := 1, 1
	for _, s := range shape[:d1] {
		pre *= s
	}
	for _, s := range shape[d2+1:] {
		post *= s
	}
	out := slices.Clone(shape)
	out[d1], out[d2] = shape[d2], shape[d1]
	return tensor.FromSlice(swapAxes(t.Data(), pre, shape[d1], shape[d2], post), out...)
}

// isSuffix reports whether small equals the trailing dims of big (the bias pattern:
// (C) against (B, T, C), or a (T, T) mask against (B, h, T, T)).
func isSuffix(big, small []int) bool {
	return len(small) <= len(big) && slices.Equal(big[len(big)-len(small):], small)
}
