package heappattern

import (
	"slices"
	"testing"
)

func TestKMostFrequentElements(t *testing.T) {
	table := []struct {
		nums []int
		k    int
		res  []int
	}{
		{
			nums: []int{1, 1, 1, 2, 2, 3, 4, 4, 4, 4, 5},
			k:    2,
			res:  []int{1, 4},
		},
	}

	for _, row := range table {
		res := kMostFrequentElements(row.nums, row.k)
		if !slices.Equal(res, row.res) {
			t.Errorf("expected %v, but got %v", row.res, res)
		}
	}
}
