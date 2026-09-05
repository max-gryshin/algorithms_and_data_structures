package heappattern

import "container/heap"

type mostFrequentElement struct {
	el   int
	freq int
}

type mostFrequentElementsHeap []mostFrequentElement

func (mf mostFrequentElementsHeap) Len() int           { return len(mf) }
func (mf mostFrequentElementsHeap) Less(i, j int) bool { return mf[i].freq < mf[j].freq }
func (mf mostFrequentElementsHeap) Swap(i, j int)      { mf[i], mf[j] = mf[j], mf[i] }

func (mf *mostFrequentElementsHeap) Pop() any {
	old := *mf
	oldLen := len(old)
	lastElem := old[oldLen-1]
	*mf = old[:oldLen-1]
	return lastElem
}
func (mf *mostFrequentElementsHeap) Push(el any) {
	*mf = append(*mf, el.(mostFrequentElement))
}

// kMostFrequentElements
// nums = [1,1,1,2,2,3,4,4,4,4,5]
// k = 2
//
// result = [4,1]
// iterate over the slice
// store k most frequent elements  in   the heap
// store frequency of the elements in   the map
// take  frequency of the element  from the map
// swap element with the least frequency in the heap if the frequency of the current element is bigger
// before checking and swaping update frequency taking actual value from the map
func kMostFrequentElements(nums []int, k int) []int {
	res := make([]int, 0, k)
	frequencyMap := make(map[int]int)
	for i := 0; i < len(nums); i++ {
		frequencyMap[nums[i]]++
	}

	mf := &mostFrequentElementsHeap{}
	heap.Init(mf)
	for num, freq := range frequencyMap {
		newElem := mostFrequentElement{el: num, freq: freq}
		if mf.Len() == k {
			if newElem.freq > (*mf)[0].freq {
				heap.Pop(mf)
			} else {
				continue
			}
		}
		heap.Push(mf, newElem)
	}

	for mf.Len() > 0 {
		res = append(res, heap.Pop(mf).(mostFrequentElement).el)
	}

	return res
}
