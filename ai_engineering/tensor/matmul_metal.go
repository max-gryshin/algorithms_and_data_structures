//go:build darwin && cgo

package tensor

/*
#cgo LDFLAGS: -framework Foundation -framework Metal -framework MetalPerformanceShaders
#include <stddef.h>
int metal_init(void);
int metal_sgemm(float *a, size_t aCap, float *b, size_t bCap, float *out, size_t outCap,
                int n, int k, int p);
*/
import "C"
import "unsafe"

// matMulMetal is the MPS kernel when the Mac has a Metal GPU, and nil otherwise
// (see matmul_noblas.go for other platforms).
var matMulMetal func(a, b, out []float32, n, k, p int)

func init() {
	if C.metal_init() == 1 {
		matMulMetal = MatMulMetal
	}
}

// MatMulMetal runs out[n×p] = a[n×k] · b[k×p] (row-major) on the GPU through
// MPSMatrixMultiplication and waits for the result. out is overwritten.
//
// Slices from PageAligned are handed to the GPU without a copy (unified memory);
// ordinary slices are copied into Metal buffers and the result is copied back.
// Every call pays a command-buffer round trip (~100 µs), so small matrices lose to the CPU.
func MatMulMetal(a, b, out []float32, n, k, p int) {
	ok := C.metal_sgemm(
		(*C.float)(unsafe.Pointer(&a[0])), C.size_t(cap(a)*4),
		(*C.float)(unsafe.Pointer(&b[0])), C.size_t(cap(b)*4),
		(*C.float)(unsafe.Pointer(&out[0])), C.size_t(cap(out)*4),
		C.int(n), C.int(k), C.int(p))
	if ok != 1 {
		panic("tensor: Metal command buffer failed")
	}
}
