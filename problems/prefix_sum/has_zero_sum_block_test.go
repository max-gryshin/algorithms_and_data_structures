package prefix_sum

import "testing"

func TestHasZeroSumBlock(t *testing.T) {
	table := []struct {
		txs  []int
		want bool
	}{
		{
			txs:  []int{4, 2, -3, 1, 6},
			want: true,
		},
		{
			txs:  []int{5, -5, 2},
			want: true,
		},
		{
			txs:  []int{1, 2, 3, 4, 5},
			want: false,
		},
		{
			txs:  []int{3, 5, -2, -3, 8},
			want: true,
		},
	}

	for _, test := range table {
		if got := hasZeroSumBlock(test.txs); got != test.want {
			t.Errorf("hasZeroSumBlock(%v) = %v, want %v", test.txs, got, test.want)
		}
	}
}
