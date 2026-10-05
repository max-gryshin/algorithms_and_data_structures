package dsu

import (
	"slices"
	"testing"
)

func TestRedundantConnection2(t *testing.T) {
	table := []struct {
		edges    [][]int
		expected []int
		name     string
	}{
		{
			name:     "in-degree 2 case 1",
			edges:    [][]int{{1, 2}, {1, 3}, {2, 3}},
			expected: []int{2, 3},
		},
		{
			name:     "in-degree 2 case 2",
			edges:    [][]int{{1, 2}, {2, 3}, {3, 4}, {4, 1}, {1, 5}},
			expected: []int{4, 1},
		},
		{
			name:     "cycle case",
			edges:    [][]int{{1, 2}, {2, 3}, {3, 4}, {4, 1}},
			expected: []int{4, 1},
		},
	}

	for _, row := range table {
		t.Run(row.name, func(t *testing.T) {
			result := redundantConnection2(row.edges)
			if !slices.Equal(result, row.expected) {
				t.Errorf("redundantConnection2(%v) = %v, expected %v", row.edges, result, row.expected)
			}
		})
	}
}
