package dfs_bfs

import (
	"slices"
	"testing"
)

func TestBfsTopologicalSort(t *testing.T) {
	table := []struct {
		graph map[int][]int
		res   []int
	}{{
		graph: map[int][]int{
			0: []int{1, 2},
			1: []int{3},
			2: []int{3},
			3: []int{4},
			4: []int{},
		},
		res: []int{0, 1, 2, 3, 4},
	}}

	for _, row := range table {
		res := kahnTopologicalSort(row.graph)
		if !slices.Equal(res, row.res) {
			t.Errorf("expected %v, but got %v", row.res, res)
		}
	}
}

func TestDfsTopologicalSort(t *testing.T) {
	table := []struct {
		graph map[int][]int
		res   []int
		err   error
	}{{
		graph: map[int][]int{
			0: []int{1, 2},
			1: []int{3},
			2: []int{3},
			3: []int{4},
			4: []int{},
		},
		res: []int{0, 2, 1, 3, 4},
	}}

	for _, row := range table {
		res, err := topologicalSort(row.graph)
		if err != row.err {
			t.Errorf("expected error %v, but got %v", row.err, err)
			return
		}
		if !slices.Equal(res, row.res) {
			t.Errorf("expected %v, but got %v", row.res, res)
		}
	}
}
