package dsu

import "testing"

func TestNumSimilarGroups(t *testing.T) {
	tests := []struct {
		name     string
		strs     []string
		expected int
	}{
		{
			name:     "single string",
			strs:     []string{"abc"},
			expected: 1,
		},
		{
			name:     "two identical strings",
			strs:     []string{"abc", "abc"},
			expected: 2,
		},
		{
			name:     "two strings one swap apart",
			strs:     []string{"abc", "bac"},
			expected: 1,
		},
		{
			name:     "two completely different strings",
			strs:     []string{"abc", "def"},
			expected: 2,
		},
		{
			name:     "two character strings one swap apart",
			strs:     []string{"ab", "ba"},
			expected: 1,
		},
		{
			name:     "two character strings different",
			strs:     []string{"ab", "cd"},
			expected: 2,
		},
		{
			name:     "group of similar strings",
			strs:     []string{"abc", "bac", "def", "fed"},
			expected: 2,
		},
		{
			name:     "all different strings",
			strs:     []string{"abc", "def", "ghi", "jkl"},
			expected: 4,
		},
		{
			name:     "three strings all different",
			strs:     []string{"aaa", "bbb", "ccc"},
			expected: 3,
		},
		{
			name:     "strings with matching char patterns",
			strs:     []string{"xyx", "yxx"},
			expected: 1,
		},
		{
			name:     "similar pair within group",
			strs:     []string{"abc", "xyz", "bac"},
			expected: 2,
		},
		{
			name:     "two groups of similar strings",
			strs:     []string{"abc", "bac", "def", "dfe"},
			expected: 2,
		},
		{
			name:     "single swap creates connection",
			strs:     []string{"aabb", "abab"},
			expected: 2,
		},
		{
			name:     "all identical",
			strs:     []string{"xxx", "xxx", "xxx"},
			expected: 1,
		},
		{
			name:     "partial similarity chain",
			strs:     []string{"abcd", "bacd", "cbad"},
			expected: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := numSimilarGroups(tt.strs)
			if got != tt.expected {
				t.Errorf("numSimilarGroups(%v) = %d, want %d", tt.strs, got, tt.expected)
			}
		})
	}
}
