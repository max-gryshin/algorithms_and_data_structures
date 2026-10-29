package tensor

import (
	"fmt"
	"runtime"
	"sync"
)

// MatMul multiplies (..., n, k) by (..., k, p) and returns (..., n, p).
// Leading (batch) dimensions are broadcast, so (B, h, T, d) x (B, h, d, T) works,
// and so does (B, T, C) x (C, D) — the same weight matrix applied to every batch element.
func MatMul(a, b *Tensor) *Tensor {
	if a.Dims() < 2 || b.Dims() < 2 {
		panic(fmt.Sprintf("tensor: MatMul needs >=2-D tensors, got %v and %v", a.shape, b.shape))
	}
	n, k := a.shape[a.Dims()-2], a.shape[a.Dims()-1]
	k2, p := b.shape[b.Dims()-2], b.shape[b.Dims()-1]
	if k != k2 {
		panic(fmt.Sprintf("tensor: MatMul inner dims differ: %v x %v", a.shape, b.shape))
	}
	batch, err := BroadcastShapes(a.shape[:a.Dims()-2], b.shape[:b.Dims()-2])
	if err != nil {
		panic(err)
	}

	a, b = a.Contiguous(), b.Contiguous()
	out := New(append(clone(batch), n, p)...)
	sa := (&Tensor{shape: a.shape[:a.Dims()-2], strides: rowMajorStrides(a.shape[:a.Dims()-2])}).broadcastStrides(batch)
	sb := (&Tensor{shape: b.shape[:b.Dims()-2], strides: rowMajorStrides(b.shape[:b.Dims()-2])}).broadcastStrides(batch)

	bi := 0
	eachIndex(batch, func(idx []int) {
		var ia, ib int
		for d, i := range idx {
			ia += i * sa[d]
			ib += i * sb[d]
		}
		am := a.data[a.offset+ia*n*k : a.offset+(ia+1)*n*k]
		bm := b.data[b.offset+ib*k*p : b.offset+(ib+1)*k*p]
		om := out.data[bi*n*p : (bi+1)*n*p]
		MatMulParallel(am, bm, om, n, k, p)
		bi++
	}) // for 2-D inputs batch is empty and this runs exactly once
	return out
}

// The kernels below work on raw row-major slices: a is n×k, b is k×p, out is n×p.
// out must be zeroed by the caller. They exist side by side for the L1 benchmark.

// MatMulNaive is the textbook i-j-k loop. Two things slow it down: the inner loop walks b
// down a column (stride p), and every iteration adds into the same accumulator s, so each
// FMA waits for the previous one to finish. MatMulNaive4 separates the two effects.
func MatMulNaive(a, b, out []float32, n, k, p int) {
	for i := range n {
		for j := range p {
			var s float32
			for l := range k {
				s += a[i*k+l] * b[l*p+j]
			}
			out[i*p+j] = s
		}
	}
}

// MatMulNaive4 keeps the column walk of MatMulNaive but splits the sum into four independent
// accumulators. It is the L1 control experiment: if it catches up with MatMulIKJ, the naive
// loop was limited by the FMA dependency chain, not by the cache. The summation order differs
// from MatMulNaive, so results agree only up to float rounding.
func MatMulNaive4(a, b, out []float32, n, k, p int) {
	for i := range n {
		for j := range p {
			var s0, s1, s2, s3 float32
			l := 0
			for ; l+4 <= k; l += 4 {
				s0 += a[i*k+l] * b[l*p+j]
				s1 += a[i*k+l+1] * b[(l+1)*p+j]
				s2 += a[i*k+l+2] * b[(l+2)*p+j]
				s3 += a[i*k+l+3] * b[(l+3)*p+j]
			}
			for ; l < k; l++ {
				s0 += a[i*k+l] * b[l*p+j]
			}
			out[i*p+j] = (s0 + s1) + (s2 + s3)
		}
	}
}

// MatMulIKJ reorders the loops so the inner loop walks rows of b and out sequentially:
// out[i,:] += a[i,l] * b[l,:]. Same arithmetic, cache-friendly access.
func MatMulIKJ(a, b, out []float32, n, k, p int) {
	for i := range n {
		orow := out[i*p : (i+1)*p]
		for l := range k {
			ail := a[i*k+l]
			brow := b[l*p : (l+1)*p]
			for j, blj := range brow {
				orow[j] += ail * blj
			}
		}
	}
}

const tile = 64

// MatMulTiled processes the matrices in tile×tile blocks so that the working set
// of the three inner loops fits in L1/L2 cache.
func MatMulTiled(a, b, out []float32, n, k, p int) {
	matMulTiledRows(a, b, out, n, k, p, 0, n)
}

func matMulTiledRows(a, b, out []float32, n, k, p, rowFrom, rowTo int) {
	for i0 := rowFrom; i0 < rowTo; i0 += tile {
		iMax := min(i0+tile, rowTo)
		for l0 := 0; l0 < k; l0 += tile {
			lMax := min(l0+tile, k)
			for j0 := 0; j0 < p; j0 += tile {
				jMax := min(j0+tile, p)
				for i := i0; i < iMax; i++ {
					orow := out[i*p+j0 : i*p+jMax]
					for l := l0; l < lMax; l++ {
						ail := a[i*k+l]
						brow := b[l*p+j0 : l*p+jMax]
						for j, blj := range brow {
							orow[j] += ail * blj
						}
					}
				}
			}
		}
	}
}

// MatMulParallel splits the rows of out into equal chunks (multiples of the tile) between at most
// GOMAXPROCS goroutines, each running the tiled kernel: for n = 128, 512, 1024 that is 2, 8 and 16
// goroutines. Rows are independent, so no synchronization on out is needed. The split is static,
// so on a CPU with cores of different speed the slowest goroutine decides the total time.
func MatMulParallel(a, b, out []float32, n, k, p int) {
	if n*k*p < 64*64*64 {
		MatMulIKJ(a, b, out, n, k, p)
		return
	}
	workers := min(runtime.GOMAXPROCS(0), (n+tile-1)/tile)
	chunk := ((n+workers-1)/workers + tile - 1) / tile * tile
	var wg sync.WaitGroup
	for from := 0; from < n; from += chunk {
		to := min(from+chunk, n)
		wg.Go(func() { matMulTiledRows(a, b, out, n, k, p, from, to) })
	}
	wg.Wait()
}
