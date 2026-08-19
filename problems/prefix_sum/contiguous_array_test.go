package prefix_sum

import "testing"

func TestFindMaxLength(t *testing.T) {
	table := []struct {
		nums     []int
		expected int
	}{
		{[]int{0, 1}, 2},
		{[]int{0, 1, 0}, 2},
		{[]int{0, 1, 1, 1, 1, 1, 0, 0, 0}, 6},
	}

	for _, tc := range table {
		actual := findMaxLength(tc.nums)
		if actual != tc.expected {
			t.Errorf("expected %d, got %d", tc.expected, actual)
		}
	}
}
