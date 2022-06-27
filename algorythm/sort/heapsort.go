package sort

func left(i int) int { return 2 * i }

func right(i int) int { return 2*i + 1 }

// Complexity O(lg n)
// needs for supporting non-increasing pyramid
func maxHeapify(a *[]int, i int, heapSize int) {
	l := left(i)
	r := right(i)
	var largest int
	// we define which element is the largest left, right or parent
	if l <= heapSize && (*a)[l-1] > (*a)[i-1] {
		largest = l
	} else {
		largest = i
	}
	if r <= heapSize && (*a)[r-1] > (*a)[largest-1] {
		largest = r
	}
	// if largest is left or right we swap values
	if largest != i {
		// complexity 0(1)
		(*a)[i-1], (*a)[largest-1] = (*a)[largest-1], (*a)[i-1]
		// to support property of non-increasing pyramid
		maxHeapify(a, largest, heapSize)
	}
}

// BuildMaxHeap building a non-increasing pyramid
// Complexity O(n) - linear time
// Property of non-increasing pyramid is that the condition is fulfilled
// for each node (except of root) with i index
// A[parent(i)] >= A[i]
func BuildMaxHeap(a *[]int, length int) {
	// we start building a pyramid from the tree leaves
	// which each of them is pyramid with one element
	for i := length / 2; i > 0; i-- {
		maxHeapify(a, i, length)
	}
}

// Complexity O(n log n)
func heapSort(a *[]int) {
	length := len(*a)
	BuildMaxHeap(a, length)
	for i := length; i > 1; i-- {
		// The largest element (A[0]) is in the top of the pyramid
		// So we swap it with the smallest (A[n]) element
		(*a)[0], (*a)[i-1] = (*a)[i-1], (*a)[0]
		// we decrement heap size
		length--
		// recovery a property of non-increasing pyramid
		maxHeapify(a, 1, length)
	}
}
