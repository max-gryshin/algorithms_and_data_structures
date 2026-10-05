package dsu

import (
	"testing"
)

func TestSmallestStringWithSwaps(t *testing.T) {
	table := []struct {
		s        string
		pairs    [][]int
		expected string
	}{
		{
			s:        "dcab",
			pairs:    [][]int{{0, 3}, {1, 2}},
			expected: "bacd",
		},
		{
			s:        "dcab",
			pairs:    [][]int{{0, 3}, {1, 2}, {0, 2}},
			expected: "abcd",
		},
		{
			s:        "cdbea",
			pairs:    [][]int{{0, 4}, {1, 3}, {1, 2}},
			expected: "abdec",
		},
		{
			s:        "a",
			pairs:    [][]int{},
			expected: "a",
		},
		{
			s:        "ba",
			pairs:    [][]int{{0, 1}},
			expected: "ab",
		},
	}

	for _, row := range table {
		result := smallestStringWithSwaps(row.s, row.pairs)
		if result != row.expected {
			t.Errorf("smallestStringWithSwaps(%q, %v) = %q, expected %q", row.s, row.pairs, result, row.expected)
		}
	}
}
