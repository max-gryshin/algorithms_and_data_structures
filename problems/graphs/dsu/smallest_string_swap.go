package dsu

import (
	"algorithms_and_data_structures/problems/graphs/dsu/theory"
	"slices"
)

// Input: s = "dcab", pairs = [[0,3],[1,2]]
// Output: "bacd"
// Explaination:
// Swap s[0] and s[3], s = "bcad"
// Swap s[1] and s[2], s = "bacd"
// Example 2:
//
// Input: s = "dcab", pairs = [[0,3],[1,2],[0,2]]
// Output: "abcd"
// Explaination:
// Swap s[0] and s[3], s = "bcad"
// Swap s[0] and s[2], s = "acbd"
// Swap s[1] and s[2], s = "abcd"
func smallestStringWithSwaps(s string, pairs [][]int) string {
	runes := []rune(s)
	dsu := theory.NewDSU(len(runes))
	dsu.BuildComponents(pairs)

	// store components by root
	components := make(map[int][]int)
	for i := range runes {
		root := dsu.FindSet(i)
		components[root] = append(components[root], i)
	}

	res := make([]rune, len(runes))
	for _, component := range components {
		// characters of the component
		subRes := []rune{}
		for _, idx := range component {
			subRes = append(subRes, runes[idx])
		}
		slices.Sort(subRes)
		// charIndex - index of the character of the original string
		// componentIndex - just var with normal order but since subRes sorted we can use it
		// to take sorted characters consequently
		for componentIndex, charIndex := range component {
			res[charIndex] = subRes[componentIndex]
		}
	}

	return string(res)
}
