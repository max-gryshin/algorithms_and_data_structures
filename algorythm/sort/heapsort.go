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
	if l <= heapSize && a[l] > a[i] {
		largest = l
	} else {
		largest = i
	}
	if r <= heapSize && a[r] > a[largest] {
		largest = r
	}
	// if largest is left or right we swap values
	if largest != i {
		// complexity 0(1)
		a[i], a[largest] = a[largest], a[i]
		// to support property of non-increasing pyramid
		MaxHeapifyRecursive(a, largest, heapSize)
	}
}

// MinHeapifyRecursive Complexity O(lg n)
// needs for supporting non-descending pyramid
// Recursive approach
func MinHeapifyRecursive(a []int, i int, heapSize int) {
	l := left(i)
	r := right(i)
	var min int
	// we define which element is the min left, right or parent
	if l <= heapSize && a[l] < a[i] {
		min = l
	} else {
		min = i
	}
	if r <= heapSize && a[r] < a[min] {
		min = r
	}
	// if min is left or right we swap values
	if min != i {
		// complexity 0(1)
		a[i], a[min] = a[min], a[i]
		// to support property of non-increasing pyramid
		MinHeapifyRecursive(a, min, heapSize)
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
		if l <= heapSize && a[l] > a[k] {
			largest = l
		} else {
			largest = k
		}
		if r <= heapSize && a[r] > a[largest] {
			largest = r
		}
		// if largest is left or right we swap values
		if largest != k {
			// complexity 0(1)
			a[k-1], a[largest] = a[largest], a[k]
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
	for i := length / 2; i >= 0; i-- {
		MaxHeapifyRecursive(a, i, length)
	}
}

func BuildMinHeap(a []int, length int) {
	// we start building a pyramid from the tree leaves
	// which each of them is pyramid with one element
	for i := length / 2; i >= 0; i-- {
		MinHeapifyRecursive(a, i, length)
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
	length := len(a) - 1
	max := a[0]
	a = a[:length+copy(a[length:], a[length+1:])]
	MaxHeapifyRecursive(a, 0, length)
	return max
}

// HeapExtractMin O(log n)
func HeapExtractMin(a *[]int) int {
	if len(*a) == 0 {
		return 0
	}
	length := len(*a) - 1
	min := (*a)[0]
	*a = (*a)[:0+copy((*a)[0:], (*a)[0+1:])]
	MinHeapifyRecursive(*a, 0, length)
	return min
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

// HeapDecreaseKey O(log n)
func HeapDecreaseKey(a []int, i, k int) ([]int, error) {
	if i >= len(a) {
		a = append(a, k)
	} else {
		if k > a[i] {
			return nil, errors.New("new key more than current")
		}
		a[i] = k
	}
	for i > 1 && a[parent(i)] < a[i] {
		a[i], a[parent(i)] = a[parent(i)], a[i]
		i = parent(i)
	}
	return a, nil
}

// HeapSort
// Complexity O(n log n)
func HeapSort(a []int) {
	length := len(a) - 1
	BuildMaxHeap(a, length)
	for i := length; i > 0; i-- {
		// The largest element (A[0]) is in the top of the pyramid
		// So we swap it with the smallest (A[n]) element
		a[0], a[i] = a[i], a[0]
		// we decrement heap size
		length--
		// recovery a property of non-increasing pyramid
		MaxHeapifyRecursive(a, 0, length)
	}
}

func HeapSortDesc(a []int) {
	length := len(a) - 1
	BuildMinHeap(a, length)
	for i := length; i > 0; i-- {
		// The largest element (A[0]) is in the top of the pyramid
		// So we swap it with the smallest (A[n]) element
		a[0], a[i] = a[i], a[0]
		// we decrement heap size
		length--
		// recovery a property of non-increasing pyramid
		MinHeapifyRecursive(a, 0, length)
	}
}
