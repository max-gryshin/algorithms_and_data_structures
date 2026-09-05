package heappattern

import "container/heap"

type kSortedHeap []int

func (h kSortedHeap) Len() int           { return len(h) }
func (h kSortedHeap) Less(i, j int) bool { return h[i] < h[j] }
func (h kSortedHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *kSortedHeap) Pop() any {
	// old is a pointer of the heap
	old := *h
	// len of the heap
	n := len(old)
	// last element
	lastElement := old[n-1]
	// remove last element from the heap
	*h = old[:n-1]
	return lastElement
}
func (h *kSortedHeap) Push(i any) { *h = append(*h, i.(int)) }

// Sort a K-Sorted Array

// Input : k = 2 , [3, 2, 1, 4, 6, 5]
func sortKSortedArr(arr []int, k int) []int {
	h := &kSortedHeap{}
	heap.Init(h)
	for i := 0; i < len(arr) && i <= k; i++ {
		heap.Push(h, arr[i])
	}
	res := make([]int, 0, len(arr))
	for i := k + 1; i < len(arr); i++ {
		res = append(res, heap.Pop(h).(int))
		heap.Push(h, arr[i])
	}
	for h.Len() > 0 {
		res = append(res, heap.Pop(h).(int))
	}

	return arr
}
