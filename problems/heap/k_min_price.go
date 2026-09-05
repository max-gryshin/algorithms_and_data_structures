package heappattern

import "container/heap"

type minPriceHeap []int

func (mph minPriceHeap) Len() int           { return len(mph) }
func (mph minPriceHeap) Less(i, j int) bool { return mph[i] > mph[j] }
func (mph minPriceHeap) Swap(i, j int)      { mph[i], mph[j] = mph[j], mph[i] }
func (mph *minPriceHeap) Push(el any)       { *mph = append(*mph, el.(int)) }
func (mph *minPriceHeap) Pop() any {
	old := *mph
	oldLen := len(old)
	lastElem := old[oldLen-1]
	*mph = old[:oldLen-1]

	return lastElem
}

// prices = [7, 10, 4, 3, 20, 15, 2, 8], k = 3
// prices = [7, 10, 3, 4, 20, 15, 2, 8], k = 3
// res = [2,3,4]
func kMinPrice(nums []int, k int) []int {
	mph := &minPriceHeap{}
	heap.Init(mph)
	for i := 0; i < len(nums); i++ {
		if mph.Len() < k {
			heap.Push(mph, nums[i])
			continue
		}
		if nums[i] < (*mph)[0] {
			heap.Pop(mph)
			heap.Push(mph, nums[i])
		}
	}
	var res = make([]int, 0, k)
	for mph.Len() > 0 {
		res = append(res, heap.Pop(mph).(int))
	}

	return res
}
