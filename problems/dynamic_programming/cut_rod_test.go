package dynamic_programming

import "testing"

func TestCutRod(t *testing.T) {
	table := []struct {
		p   map[int]int
		n   int
		res int
	}{
		{
			p: map[int]int{
				1:  1,
				2:  5,
				3:  8,
				4:  9,
				5:  10,
				6:  17,
				7:  17,
				8:  20,
				9:  24,
				10: 30,
			},
			n:   10,
			res: 30,
		},
	}

	for _, row := range table {
		res := cutRod(row.p, row.n)
		if res != row.res {
			t.Errorf("expected %d, but got %d", row.res, res)
		}
	}
}
