package problems

import "testing"

var LongestCommonPrefixData = []struct {
	n        []string
	expected string
}{
	{
		[]string{"flower", "flow", "flight"},
		"fl",
	},
}

func TestLongestCommonPrefix(t *testing.T) {
	for _, v := range LongestCommonPrefixData {
		res := longestCommonPrefix(v.n)
		if v.expected != res {
			t.Errorf(
				" the result of longestCommonPrefix(%v) not correct: expected %s, actual %s",
				v.n,
				v.expected,
				res,
			)
		}
	}
}
