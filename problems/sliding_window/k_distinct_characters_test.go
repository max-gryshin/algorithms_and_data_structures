package sliding_window

import "testing"

func TestKDistinctCharacters(t *testing.T) {
	table := []struct {
		nums string
		k    int
		res  int
	}{
		{
			nums: "eceba",
			k:    2,
			res:  3,
		},
	}

	for _, row := range table {
		res := kDistinctCharacters(row.nums, row.k)
		if res != row.res {
			t.Errorf("expect %v but got %v", row.res, res)
		}
	}
}
