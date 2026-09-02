package prefix_sum

import "testing"

func TestNumSubarraysWithSum(t *testing.T) {
	table := []struct {
		nums     []int
		goal     int
		expected int
	}{
		{
			nums:     []int{1, 0, 1, 0, 1},
			goal:     2,
			expected: 4,
		},
		{
			nums:     []int{0, 0, 0, 0, 0},
			goal:     0,
			expected: 15,
		},
	}

	for _, row := range table {
		res := numSubarraysWithSum(row.nums, row.goal)
		if res != row.expected {
			t.Errorf("error expected %d, but got %d", row.expected, res)
		}
	}
}
