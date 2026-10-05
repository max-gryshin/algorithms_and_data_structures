package dsu

import (
	"algorithms_and_data_structures/problems/graphs/dsu/theory"
)

func redundantConnection(edges [][]int) []int {
	dsu := theory.NewDSU(len(edges) + 1)
	for _, edge := range edges {
		x, y := edge[0], edge[1]
		px, py := dsu.FindSet(x), dsu.FindSet(y)
		if px == py {
			return []int{x, y}
		}
		dsu.Union(x, y)
	}

	return nil
}
