package problems

import "testing"

func TestFindSimilar(t *testing.T) {
	table := []struct {
		a   string
		b   string
		res int
	}{
		{
			a:   "1234",
			b:   "2341",
			res: 24,
		},
		{
			a:   "11223344",
			b:   "44332211",
			res: 2520,
		},
		{
			a:   "1001223344",
			b:   "9988776655",
			res: 113400,
		},
		{
			a:   "1234567891",
			b:   "9876543211",
			res: 1814400,
		},
		{
			a:   "100020",
			b:   "200001",
			res: 10,
		},
	}

	for _, row := range table {
		res := findSimilar(row.a, row.b)
		if res != row.res {
			t.Errorf("expected %d, but got %d", row.res, res)
		}
	}
}
