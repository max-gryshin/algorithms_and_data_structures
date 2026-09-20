package dfs_bfs

import (
	"slices"
	"testing"
)

func TestFindSCCKosaraju(t *testing.T) {
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
				{1, 2, 3}, {4}, {5},
			},
		},
	}

	for _, row := range table {
		edges := findSCCKosaraju(row.v, row.edges)
		for i := 0; i < len(row.scc); i++ {
			if !slices.Equal(edges[i], row.scc[i]) {
				t.Errorf("expected %v, but got %v", row.scc, edges)
			}
		}
	}
}
