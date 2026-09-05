package heappattern

import (
	"slices"
	"testing"
)

// prices = [7, 10, 4, 3, 20, 15, 2, 8], k = 3
// res = [2,3,4]
func TestKMinPrice(t *testing.T) {
	table := []struct {
		nums []int
		k    int
		res  []int
	}{
		{
			nums: []int{7, 10, 4, 3, 20, 15, 2, 8},
			k:    3,
			res:  []int{4, 3, 2},
		},
	}

	for _, row := range table {
		res := kMinPrice(row.nums, row.k)
		if !slices.Equal(res, row.res) {
			t.Errorf("expected %v, but got %v", row.res, res)
		}
	}
}
