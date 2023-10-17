package sort

import (
	"math"
	"math/rand"
	"reflect"
	"testing"
)

type heapSliceInt []int

func (h *heapSliceInt) Less(i, j int) bool {
	return (*h)[i] < (*h)[j]
}
func (h *heapSliceInt) Swap(i, j int) {
	(*h)[i], (*h)[j] = (*h)[j], (*h)[i]
}
func (h *heapSliceInt) Len() int {
	return len(*h)
}

var heapTests = []struct {
	n        heapSliceInt
	expected heapSliceInt
}{
	{
		heapSliceInt{4, 1, 3, 2, 16, 9, 10, 14, 8, 7},
		[]int{1, 2, 3, 4, 7, 8, 9, 10, 14, 16},
	},
	{
		rand.Perm(10),
		makeRange(0, 9),
	},
	{
		rand.Perm(100),
		makeRange(0, 99),
	},
	{
		rand.Perm(int(math.Pow(2, 10))),
		makeRange(0, int(math.Pow(2, 10))-1),
	},
	{
		rand.Perm(int(math.Pow(2, 16))),
		makeRange(0, int(math.Pow(2, 16))-1),
	},
	{
		rand.Perm(int(math.Pow(2, 28))),
		makeRange(0, int(math.Pow(2, 28))-1),
	},
	// 2 ^ 28 takes about 3 minutes seconds with HeapSort(h.n) recursive approach
}

func TestHeapSort(t *testing.T) {
	for _, h := range heapTests {
		startOrder := make(heapSliceInt, len(h.n))
		copy(startOrder, h.n)
		HeapSort(&h.n)
		if !reflect.DeepEqual(h.n, h.expected) {
			t.Errorf("HeapSort(%v): expected %v, actual %v ", startOrder, h.expected, h.n)
		}
	}
}
