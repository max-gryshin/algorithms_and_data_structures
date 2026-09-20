package dfs_bfs

import (
	"slices"
	"testing"
)

func TestFindSCC(t *testing.T) {
	table := []struct {
		edges [][]int
		v     int
		scc   [][]int
	}{
		{
			edges: [][]int{
				{1, 3}, {1, 4}, {2, 1}, {3, 2}, {4, 5},
			},
			v: 5,
			scc: [][]int{
				{5}, {4}, {2, 3, 1},
			},
		},
	}

	for _, row := range table {
		sccFinder := NewTarjanSCC(row.v, row.edges)
		sccs := sccFinder.FindSCC()
		for i := 0; i < len(row.scc); i++ {
			if !slices.Equal(sccs[i], row.scc[i]) {
				t.Errorf("expected %v, but got %v", row.scc, sccs)
			}
		}
	}
}
