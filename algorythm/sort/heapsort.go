package sort

import "errors"

func left(i int) int { return 2 * i }

func right(i int) int { return 2*i + 1 }

func parent(i int) int { return i / 2 }

// MaxHeapifyRecursive Complexity O(lg n)
// needs for supporting non-increasing pyramid
// Recursive approach
func MaxHeapifyRecursive(a []int, i int, heapSize int) {
	l := left(i)
	r := right(i)
	var largest int
	// we define which element is the largest left, right or parent
	if l <= heapSize && a[l-1] > a[i-1] {
		largest = l
	} else {
		largest = i
	}
	if r <= heapSize && a[r-1] > a[largest-1] {
		largest = r
	}
	// if largest is left or right we swap values
	if largest != i {
		// complexity 0(1)
		a[i-1], a[largest-1] = a[largest-1], a[i-1]
		// to support property of non-increasing pyramid
		MaxHeapifyRecursive(a, largest, heapSize)
	}
}

// MaxHeapify Complexity O(lg n)
// needs for supporting non-increasing pyramid
func MaxHeapify(a []int, i int, heapSize int) {
	var l, r, largest int
	for k := i; k <= heapSize; k++ {
		l = left(k)
		r = right(k)
		// we define which element is the largest left, right or parent
		if l <= heapSize && a[l-1] > a[k-1] {
			largest = l
		} else {
			largest = k
		}
		if r <= heapSize && a[r-1] > a[largest-1] {
			largest = r
		}
		// if largest is left or right we swap values
		if largest != k {
			// complexity 0(1)
			a[k-1], a[largest-1] = a[largest-1], a[k-1]
			// to support property of non-increasing pyramid
		}
	}
}

// BuildMaxHeap building a non-increasing pyramid
// Complexity O(n) - linear time
// Property of non-increasing pyramid is that the condition is fulfilled
// for each node (except of root) with i index
// A[parent(i)] >= A[i]
func BuildMaxHeap(a []int, length int) {
	// we start building a pyramid from the tree leaves
	// which each of them is pyramid with one element
	for i := length / 2; i > 0; i-- {
		MaxHeapifyRecursive(a, i, length)
	}
}

// MaxHeapInsert  O(log n)
func MaxHeapInsert(a []int, key int) ([]int, error) {
	return HeapIncreaseKey(a, len(a), key)
}

// HeapExtractMax O(log n)
func HeapExtractMax(a []int) int {
	if len(a) == 0 {
		return 0
	}
	aLen := len(a) - 1
	max := a[0]
	a[0] = a[aLen]
	MaxHeapify(a, 1, aLen)
	return max
}

// HeapIncreaseKey O(log n)
func HeapIncreaseKey(a []int, i, k int) ([]int, error) {
	if i >= len(a) {
		a = append(a, k)
	} else {
		if k < a[i] {
			return nil, errors.New("new key less than current")
		}
		a[i] = k
	}
	for i > 0 && a[parent(i)] < a[i] {
		a[i], a[parent(i)] = a[parent(i)], a[i]
		i = parent(i)
	}
	return a, nil
}

// HeapSort
// Complexity O(n log n)
func HeapSort(a []int) {
	length := len(a)
	BuildMaxHeap(a, length)
	for i := length; i > 1; i-- {
		// The largest element (A[0]) is in the top of the pyramid
		// So we swap it with the smallest (A[n]) element
		a[0], a[i-1] = a[i-1], a[0]
		// we decrement heap size
		length--
		// recovery a property of non-increasing pyramid
		MaxHeapifyRecursive(a, 1, length)
	}
}
