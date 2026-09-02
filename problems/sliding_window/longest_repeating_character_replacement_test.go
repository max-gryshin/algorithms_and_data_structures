package sliding_window

import "testing"

func TestLongestRepeatingCharacterReplacement(t *testing.T) {
	table := []struct {
		s        string
		k        int
		expected int
	}{
		{
			s:        "AABABBA",
			k:        1,
			expected: 4,
		},
		{
			s:        "CCCCAABABBA",
			k:        1,
			expected: 5,
		},
	}

	for _, row := range table {
		res := LongestRepeatingCharacterReplacement(row.s, row.k)
		if res != row.expected {
			t.Errorf("error expected %d, but got %d", row.expected, res)
		}
	}
}
