package sort

type Heap interface {
	Less(i, j int) bool
	Swap(i, j int)
	Len() int
}

func left(i int) int { return 2 * i }

func right(i int) int { return 2*i + 1 }

func parent(i int) int { return i / 2 }

// MaxHeapifyRecursive Complexity O(lg n)
// needs for supporting non-increasing pyramid
// Recursive approach
func MaxHeapifyRecursive(h Heap, i int, heapSize int) {
	l := left(i)
	r := right(i)
	var largest int
	// we define which element is the largest left, right or parent
	if l <= heapSize && h.Less(i, l) {
		largest = l
	} else {
		largest = i
	}
	if r <= heapSize && h.Less(largest, r) {
		largest = r
	}
	// if largest is left or right we swap values
	if largest != i {
		// complexity 0(1)
		h.Swap(i, largest)
		// to support property of non-increasing pyramid
		MaxHeapifyRecursive(h, largest, heapSize)
	}
}

// MinHeapifyRecursive Complexity O(lg n)
// needs for supporting non-descending pyramid
// Recursive approach
func MinHeapifyRecursive(h Heap, i int, heapSize int) {
	l := left(i)
	r := right(i)
	var min int
	// we define which element is the min left, right or parent
	if l <= heapSize && h.Less(l, i) {
		min = l
	} else {
		min = i
	}
	if r <= heapSize && h.Less(r, min) {
		min = r
	}
	// if min is left or right we swap values
	if min != i {
		// complexity 0(1)
		h.Swap(i, min)
		// to support property of non-increasing pyramid
		MinHeapifyRecursive(h, min, heapSize)
	}
}

// MaxHeapify Complexity O(lg n)
// needs for supporting non-increasing pyramid
func MaxHeapify(h Heap, i int, heapSize int) {
	var l, r, largest int
	for k := i; k <= heapSize; k++ {
		l = left(k)
		r = right(k)
		// we define which element is the largest left, right or parent
		if l <= heapSize && h.Less(k, l) {
			largest = l
		} else {
			largest = k
		}
		if r <= heapSize && h.Less(largest, r) {
			largest = r
		}
		// if largest is left or right we swap values
		if largest != k {
			// complexity 0(1)
			h.Swap(k-1, largest)
			// to support property of non-increasing pyramid
		}
	}
}

// BuildMaxHeap building a non-increasing pyramid
// Complexity O(n) - linear time
// Property of non-increasing pyramid is that the condition is fulfilled
// for each node (except of root) with i index
// A[parent(i)] >= A[i]
func BuildMaxHeap(h Heap, length int) {
	// we start building a pyramid from the tree leaves
	// which each of them is pyramid with one element
	for i := length / 2; i >= 0; i-- {
		MaxHeapifyRecursive(h, i, length)
	}
}

func BuildMinHeap(h Heap, length int) {
	// we start building a pyramid from the tree leaves
	// which each of them is pyramid with one element
	for i := length / 2; i >= 0; i-- {
		MinHeapifyRecursive(h, i, length)
	}
}

// HeapSort
// Complexity O(n log n)
func HeapSort(h Heap) {
	length := h.Len() - 1
	BuildMaxHeap(h, length)
	for i := length; i > 0; i-- {
		// The largest element (A[0]) is in the top of the pyramid
		// So we swap it with the smallest (A[n]) element
		h.Swap(0, i)
		// we decrement heap size
		length--
		// recovery a property of non-increasing pyramid
		MaxHeapifyRecursive(h, 0, length)
	}
}

func HeapSortDesc(h Heap) {
	length := h.Len() - 1
	BuildMinHeap(h, length)
	for i := length; i > 0; i-- {
		// The largest element (A[0]) is in the top of the pyramid
		// So we swap it with the smallest (A[n]) element
		h.Swap(0, i)
		// we decrement heap size
		length--
		// recovery a property of non-increasing pyramid
		MinHeapifyRecursive(h, 0, length)
	}
}
