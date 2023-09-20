package sort

import (
	"math"
	"math/rand"
	"reflect"
	"testing"
)

var heapTests = []struct {
	n        []int
	expected []int
}{
	{
		[]int{4, 1, 3, 2, 16, 9, 10, 14, 8, 7},
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
	// 2 ^ 28 takes about 2 minutes seconds with HeapSort(&h.n) recursive approach
}

func TestHeapSort(t *testing.T) {
	for _, h := range heapTests {
		startOrder := make([]int, len(h.n))
		copy(startOrder, h.n)
		HeapSort(&h.n)
		if !reflect.DeepEqual(h.n, h.expected) {
			t.Errorf("HeapSort(%v): expected %v, actual %v ", startOrder, h.expected, h.n)
		}
	}
}
