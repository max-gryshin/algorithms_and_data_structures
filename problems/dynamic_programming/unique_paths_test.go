package dynamic_programming

import "testing"

func Test_uniquePaths(t *testing.T) {
	table := []struct {
		m        int
		n        int
		expected int
	}{
		{m: 3, n: 7, expected: 28},
		{m: 3, n: 2, expected: 3},
		{m: 3, n: 3, expected: 6},
	}

	for _, testCase := range table {
		if res := uniquePaths(testCase.m, testCase.n); res != testCase.expected {
			t.Errorf("Expected %d, got %d", testCase.expected, res)
		}
	}
}
