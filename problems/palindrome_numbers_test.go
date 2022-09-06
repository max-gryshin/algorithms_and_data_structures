package problems

import "testing"

var palindromes = []struct {
	n        int
	expected bool
}{
	{
		1001,
		true,
	},
	{
		10,
		false,
	},
	{
		-121,
		false,
	},
	{
		121,
		true,
	},
	{
		1234321,
		true,
	},
	{
		123,
		false,
	},
	{
		343,
		true,
	},
	{
		11,
		true,
	},
	{
		2332,
		true,
	},
}

func TestPalindromeNumbers(t *testing.T) {
	for _, v := range palindromes {
		res := isPalindrome(v.n)
		if res != v.expected {
			t.Errorf(
				" the result of isPalindrome(%v) not correct: expected %v, actual %v",
				v.n,
				v.expected,
				res,
			)
		}
	}
}
