package sort

import (
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
}

func TestHeapSort(t *testing.T) {
	for _, h := range heapTests {
		startOrder := make([]int, len(h.n))
		copy(startOrder, h.n)
		heapSort(&h.n)
		if !reflect.DeepEqual(h.n, h.expected) {
			t.Errorf("heapSort(%v): expected %v, actual %v ", startOrder, h.expected, h.n)
		}
	}
}
