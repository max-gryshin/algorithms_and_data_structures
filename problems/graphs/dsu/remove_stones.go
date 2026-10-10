package dsu

import "algorithms_and_data_structures/problems/graphs/dsu/theory"

func removeStones(stones [][]int) int {
	mapX := make(map[int]int)
	mapY := make(map[int]int)
	dsu := theory.NewDSU(len(stones))
	for idx, stone := range stones {
		if idxStored, ok := mapX[stone[0]]; ok {
			dsu.Union(idxStored, idx)
		} else {
			mapX[stone[0]] = idx
		}
		if idxStored, ok := mapY[stone[1]]; ok {
			dsu.Union(idxStored, idx)
		} else {
			mapY[stone[1]] = idx
		}
	}
	componentsByRoot := make(map[int]bool)
	for idx := range stones {
		componentsByRoot[dsu.FindSet(idx)] = true
	}
	return len(stones) - len(componentsByRoot)
}
