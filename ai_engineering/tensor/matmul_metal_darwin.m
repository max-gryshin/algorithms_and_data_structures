//go:build darwin && cgo

// Objective-C side of MatMulMetal (see matmul_metal.go). Built without ARC, so every
// object made by alloc/new is released by hand; autoreleased ones die with the pool.

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <string.h>
#include <unistd.h>

static id<MTLDevice> device;     // the GPU; lives for the whole process
static id<MTLCommandQueue> queue; // one queue is enough: commands run in submit order

int metal_init(void) {
	device = MTLCreateSystemDefaultDevice();
	if (device == nil || !MPSSupportsMTLDevice(device)) {
		return 0;
	}
	queue = [device newCommandQueue];
	return queue != nil;
}

// wrap returns a shared (CPU+GPU visible) buffer for p. Unified memory lets the GPU read
// p directly, but newBufferWithBytesNoCopy wants a page-aligned pointer and a length that is
// a whole number of pages. If p qualifies (its capacity covers len rounded up to a page),
// the buffer aliases p and *aliased = 1. Otherwise the data is copied into a new buffer
// (or, with copyIn = 0, a new buffer is only allocated: used for the output).
static id<MTLBuffer> wrap(void *p, size_t len, size_t cap, int copyIn, int *aliased) {
	size_t page = (size_t)getpagesize(); // 16 KiB on Apple Silicon
	size_t rounded = (len + page - 1) / page * page;
	if ((uintptr_t)p % page == 0 && cap >= rounded) {
		*aliased = 1;
		return [device newBufferWithBytesNoCopy:p length:rounded
		                                options:MTLResourceStorageModeShared deallocator:nil];
	}
	*aliased = 0;
	if (copyIn) {
		return [device newBufferWithBytes:p length:len options:MTLResourceStorageModeShared];
	}
	return [device newBufferWithLength:len options:MTLResourceStorageModeShared];
}

static MPSMatrix *matrix(id<MTLBuffer> buf, int rows, int cols) {
	MPSMatrixDescriptor *d = [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols
	                                                              rowBytes:cols * sizeof(float)
	                                                              dataType:MPSDataTypeFloat32];
	return [[MPSMatrix alloc] initWithBuffer:buf descriptor:d];
}

// metal_sgemm computes out[n×p] = a[n×k] · b[k×p] (row-major) on the GPU and blocks until
// the result is in out. The *Cap arguments are the slice capacities in bytes. Returns 1 on success.
int metal_sgemm(float *a, size_t aCap, float *b, size_t bCap, float *out, size_t outCap,
                int n, int k, int p) {
	@autoreleasepool {
		size_t aLen = (size_t)n * k * sizeof(float);
		size_t bLen = (size_t)k * p * sizeof(float);
		size_t outLen = (size_t)n * p * sizeof(float);
		int aliasA, aliasB, aliasOut;
		id<MTLBuffer> bufA = wrap(a, aLen, aCap, 1, &aliasA);
		id<MTLBuffer> bufB = wrap(b, bLen, bCap, 1, &aliasB);
		id<MTLBuffer> bufOut = wrap(out, outLen, outCap, 0, &aliasOut);

		MPSMatrix *A = matrix(bufA, n, k);
		MPSMatrix *B = matrix(bufB, k, p);
		MPSMatrix *C = matrix(bufOut, n, p);
		// C = alpha·A·B + beta·C with beta = 0, so out is overwritten like cblas_sgemm.
		MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc] initWithDevice:device
		    transposeLeft:NO transposeRight:NO resultRows:n resultColumns:p interiorColumns:k
		    alpha:1.0 beta:0.0];

		id<MTLCommandBuffer> cb = [queue commandBuffer];
		[mm encodeToCommandBuffer:cb leftMatrix:A rightMatrix:B resultMatrix:C];
		[cb commit];
		[cb waitUntilCompleted]; // the GPU is asynchronous; Go must not touch out before this
		int ok = cb.status == MTLCommandBufferStatusCompleted;
		if (ok && !aliasOut) {
			memcpy(out, bufOut.contents, outLen);
		}

		[mm release];
		[A release];
		[B release];
		[C release];
		[bufA release];
		[bufB release];
		[bufOut release];
		return ok;
	}
}
