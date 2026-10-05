package dsu

import (
	"algorithms_and_data_structures/problems/graphs/dsu/theory"
)

func findCircleNum(isConnected [][]int) int {
	n := len(isConnected)
	dsu := theory.NewDSU(len(isConnected))
	res := 0

	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if isConnected[i][j] == 1 {
				dsu.Union(i, j)
			}
		}
	}

	for i := 0; i < n; i++ {
		if dsu.FindSet(i) == i {
			res++
		}
	}

	return res
}
