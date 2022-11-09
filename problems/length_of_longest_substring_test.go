package problems

import (
	"testing"
)

var LengthOfLongestSubstringData = []struct {
	s string
	n int
}{
	{
		"ckilbkd",
		5,
	},
	{
		"abcabcbb",
		3,
	},
	{
		"pwwkew",
		3,
	},
	{
		"dvdf",
		3,
	},
}

func TestLengthOfLongestSubstring(t *testing.T) {
	for _, i := range LengthOfLongestSubstringData {
		actual := lengthOfLongestSubstring(i.s)
		if actual != i.n {
			t.Errorf(
				" the result of lengthOfLongestSubstring(%v) not correct: expected %v, actual %v",
				i.s,
				i.n,
				actual,
			)
		}
	}
}
