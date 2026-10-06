package dsu

import (
	"algorithms_and_data_structures/problems/graphs/dsu/theory"
	"slices"
)

func numSimilarGroups(strs []string) int {
	components := len(strs)
	dsu := theory.NewDSU(components)
	indexByString := make(map[string]int)
	for id, anagram := range strs {
		indexByString[anagram] = id
	}
	for id, str := range strs {
		runes := []rune(str)
	loop:
		for i := 0; i < len(runes)-1; i++ {
			for j := i + 1; j < len(runes); j++ {
				runesClone := slices.Clone(runes)
				runesClone[i], runesClone[j] = runesClone[j], runesClone[i]
				if foundID, ok := indexByString[string(runesClone)]; ok {
					if dsu.FindSet(id) != dsu.FindSet(foundID) {
						components--
					}
					dsu.Union(foundID, id)
					break loop
				}
			}
		}
	}
	return components
}
