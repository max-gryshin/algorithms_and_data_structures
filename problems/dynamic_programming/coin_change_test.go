package dynamic_programming

import "testing"

func TestCoinChange(t *testing.T) {
	table := []struct {
		coins  []int
		amount int
		res    int
	}{
		{
			coins:  []int{1, 2, 5},
			amount: 11,
			res:    3,
		},
	}

	for _, row := range table {
		res := coinChange(row.coins, row.amount)

		if res != row.res {
			t.Errorf("expected %d, but got %d", row.res, res)
		}
	}
}
