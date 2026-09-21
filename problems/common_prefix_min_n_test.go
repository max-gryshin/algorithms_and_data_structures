package problems

import (
	"testing"
)

func TestCommonPrefix(t *testing.T) {
	table := []struct {
		s    []string
		minN int
		res  string
	}{
		{
			s:    []string{"12345", "12367", "12499", "98765"},
			minN: 2,
			res:  "123",
		},
		{
			s:    []string{"987123", "987456", "987999", "987000"},
			minN: 3,
			res:  "987",
		},
		{
			s:    []string{"456789", "123456", "987654"},
			minN: 1,
			res:  "123456",
		},
		{
			s:    []string{"11111", "22222", "33333", "33344"},
			minN: 2,
			res:  "333",
		},
		{
			s:    []string{"12345", "12399", "45678", "45699"},
			minN: 2,
			res:  "123",
		},
		{
			s:    []string{"123", "12345", "12399", "98765"},
			minN: 2,
			res:  "123",
		},
		{
			s:    []string{"12345", "45678", "78901"},
			minN: 2,
			res:  "",
		},
		{
			s:    []string{"12345", "12367"},
			minN: 3,
			res:  "",
		},
		{
			s:    []string{"12345", "12345", "12399", "98765"},
			minN: 3,
			res:  "123",
		},
		{
			s: []string{
				"1029384756",
				"1029384766",
				"1029384799",
				"1029390000",
				"5555555555",
				"5555555566",
				"5559999999",
				"9999999999",
			},
			minN: 3,
			res:  "10293847",
		},
	}

	for _, row := range table {
		res := commonPrefix(row.s, row.minN)
		if res != row.res {
			t.Errorf("expected %s, but got %s", row.res, res)
		}
	}

}
