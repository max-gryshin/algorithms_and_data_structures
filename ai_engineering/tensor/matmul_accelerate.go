//go:build darwin && cgo

package tensor

/*
#cgo CFLAGS: -DACCELERATE_NEW_LAPACK
#cgo LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>
*/
import "C"
import "unsafe"

// matMulBLAS is the Accelerate kernel on darwin and nil elsewhere (see matmul_noblas.go).
var matMulBLAS = MatMulAccelerate

// MatMulAccelerate calls cblas_sgemm from Apple Accelerate: out[n×p] = a[n×k] · b[k×p], row-major.
// On Apple Silicon sgemm runs on the AMX/SME matrix units. beta = 0, so out is overwritten.
func MatMulAccelerate(a, b, out []float32, n, k, p int) {
	C.cblas_sgemm(C.CblasRowMajor, C.CblasNoTrans, C.CblasNoTrans,
		C.__LAPACK_int(n), C.__LAPACK_int(p), C.__LAPACK_int(k),
		1, (*C.float)(unsafe.Pointer(&a[0])), C.__LAPACK_int(k),
		(*C.float)(unsafe.Pointer(&b[0])), C.__LAPACK_int(p),
		0, (*C.float)(unsafe.Pointer(&out[0])), C.__LAPACK_int(p))
}
