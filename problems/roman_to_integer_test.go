package problems

import "testing"

var data = []struct {
	n        string
	expected int
}{
	{
		"DCXXI",
		621,
	},
	{
		"MCMXCIV",
		1994,
	},
	{
		"IV",
		4,
	},
	{
		"III",
		3,
	},
	{
		"IX",
		9,
	},
	{
		"LVIII",
		58,
	},
}

func TestRomanToInt(t *testing.T) {
	for _, v := range data {
		res := romanToInt(v.n)
		if v.expected != res {
			t.Errorf(
				" the result of romanToInt(%v) not correct: expected %v, actual %v",
				v.n,
				v.expected,
				res,
			)
		}
	}
}
