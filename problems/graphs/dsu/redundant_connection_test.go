package dsu

import (
	"slices"
	"testing"
)

func TestRedundantConnection(t *testing.T) {
	table := []struct {
		edges    [][]int
		expected []int
	}{
		{
			edges:    [][]int{{1, 2}, {1, 3}, {2, 3}},
			expected: []int{2, 3},
		},
		{
			edges:    [][]int{{1, 2}, {2, 3}, {3, 4}, {1, 4}},
			expected: []int{1, 4},
		},
		{
			edges:    [][]int{{1, 2}, {2, 3}, {3, 1}},
			expected: []int{3, 1},
		},
		{
			edges:    [][]int{{1, 2}, {1, 3}, {2, 4}, {3, 4}},
			expected: []int{3, 4},
		},
		{
			edges:    [][]int{{1, 2}, {2, 3}, {1, 3}},
			expected: []int{1, 3},
		},
		{
			edges:    [][]int{{1, 2}, {1, 3}, {1, 4}, {2, 5}, {2, 6}, {3, 5}, {3, 6}, {4, 5}, {4, 6}, {5, 6}},
			expected: []int{3, 5},
		},
	}

	for _, row := range table {
		result := redundantConnection(row.edges)
		if !slices.Equal(result, row.expected) {
			t.Errorf("redundantConnection(%v) = %v, expected %v", row.edges, result, row.expected)
		}
	}
}
