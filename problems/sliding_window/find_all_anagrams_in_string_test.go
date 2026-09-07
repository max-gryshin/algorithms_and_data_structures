package sliding_window

import (
	"slices"
	"testing"
)

// Example 1:
//
// Input: s = "cbaebabacd", p = "abc"
// Output: [0,6]
// Explanation:
// The substring with start index = 0 is "cba", which is an anagram of "abc".
// The substring with start index = 6 is "bac", which is an anagram of "abc".
// Example 2:
//
// Input: s = "abab", p = "ab"
// Output: [0,1,2]
// Explanation:
// The substring with start index = 0 is "ab", which is an anagram of "ab".
// The substring with start index = 1 is "ba", which is an anagram of "ab".
// The substring with start index = 2 is "ab", which is an anagram of "ab".
func TestFindAnagrams(t *testing.T) {
	table := []struct {
		s   string
		p   string
		res []int
	}{
		{
			s:   "cbaebabacd",
			p:   "abc",
			res: []int{0, 6},
		},
		{
			s:   "abab",
			p:   "ab",
			res: []int{0, 1, 2},
		},
	}

	for _, row := range table {
		res := findAnagrams(row.s, row.p)
		if !slices.Equal(res, row.res) {
			t.Errorf("expected %v but got %v", row.res, res)
		}
	}
}
