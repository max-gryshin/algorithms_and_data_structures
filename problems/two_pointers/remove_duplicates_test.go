package two_pointers

import "testing"

func TestRemoveDuplicatesSorted(t *testing.T) {
	table := []struct {
		nums     []int
		expected int
	}{
		{
			nums:     []int{1, 1, 1, 2, 2, 3},
			expected: 5,
		},
		{
			nums:     []int{0, 0, 1, 1, 1, 1, 2, 3, 3},
			expected: 7,
		},
		{
			nums:     []int{0, 0, 0, 0, 1, 1, 1, 2, 2, 2},
			expected: 6,
		},
		{
			nums:     []int{1, 1, 1, 1, 1},
			expected: 2,
		},
		{
			nums:     []int{1, 2, 3, 4, 5},
			expected: 5,
		},
		{
			nums:     []int{1},
			expected: 1,
		},
		{
			nums:     []int{1, 1, 2, 2, 3, 3, 4, 4},
			expected: 8,
		},

		{
			nums:     []int{1, 1, 1, 2, 2, 2, 3, 3, 3},
			expected: 6,
		},
		{
			nums:     []int{-3, -3, -3, -2, -2, -1},
			expected: 5,
		},
		{
			nums:     []int{-5, -5, -5, -5, -2, -2, 0, 0, 0, 1, 1, 1, 1, 3},
			expected: 9,
		},
		{
			nums:     []int{},
			expected: 0,
		},
	}

	for _, row := range table {
		res := removeDuplicatesSorted(row.nums)
		if res != row.expected && len(row.nums) != row.expected {
			t.Errorf("expected %v, but got %v", row.expected, res)
		}
	}
}
