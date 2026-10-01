package dsu

import (
	"algorithms_and_data_structures/problems/graphs/dsu/theory"
)

// todo: 685. Redundant Connection II
func redundantConnection(edges [][]int) []int {
	n := len(edges)

	dsu := theory.NewDSU(n + 1)
	for _, edge := range edges {
		u, v := edge[0], edge[1]
		if !dsu.Union(u, v) {
			return []int{u, v}
		}
	}

	return nil
}
