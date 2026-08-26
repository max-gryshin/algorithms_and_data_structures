package prefix_sum

import (
	"slices"
	"testing"
)

func TestProductExceptSelf(t *testing.T) {
	table := []struct {
		nums   []int
		answer []int
	}{
		{
			nums:   []int{1, 2, 3, 4},
			answer: []int{24, 12, 8, 6},
		},
		{
			nums:   []int{-1, 1, 0, -3, 3},
			answer: []int{0, 0, 9, 0, 0},
		},
	}

	for _, testCase := range table {
		answer := productExceptSelf(testCase.nums)
		if !slices.Equal(answer, testCase.answer) {
			t.Errorf("error: expect %v, but got %v", testCase.answer, answer)
		}
	}
}
