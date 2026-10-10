package hashmap

import (
	"container/heap"
)

type Item struct {
	num, count int
}

type ItemsHeap []Item

func (itms ItemsHeap) Len() int {
	return len(itms)
}

func (itms ItemsHeap) Less(i, j int) bool {
	return itms[i].count < itms[j].count
}

func (itms ItemsHeap) Swap(i, j int) {
	itms[i], itms[j] = itms[j], itms[i]
}

func (itms *ItemsHeap) Pop() any {
	old := *itms
	n := len(old)
	last := old[n-1]
	*itms = old[:n-1]

	return last
}

func (itms *ItemsHeap) Push(v any) {
	*itms = append(*itms, v.(Item))
}

// O(n+u log k) - time
// O(u+k) - space
func topKFreqElems(elems []int, k int) []int {
	freq := make(map[int]int)
	for _, e := range elems {
		freq[e]++
	}
	itemsHeap := &ItemsHeap{}
	heap.Init(itemsHeap)
	for num, count := range freq {
		heap.Push(itemsHeap, Item{num: num, count: count})

		if itemsHeap.Len() > k {
			heap.Pop(itemsHeap)
		}
	}
	res := make([]int, 0, k)
	for itemsHeap.Len() > 0 {
		res = append(res, heap.Pop(itemsHeap).(Item).num)
	}

	return res
}
