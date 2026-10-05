package dsu

import "algorithms_and_data_structures/problems/graphs/dsu/theory"

func redundantConnection2(edges [][]int) []int {
	n := len(edges)
	inDegree := make([]int, n+1)
	parent := make(map[int][][]int)

	// Track all edges and find nodes with in-degree 2
	for _, edge := range edges {
		inDegree[edge[1]]++
		parent[edge[1]] = append(parent[edge[1]], edge)
	}

	// Find the node with in-degree 2
	var node2 int
	for i := 1; i <= n; i++ {
		if inDegree[i] == 2 {
			node2 = i
			break
		}
	}

	// Helper function to check if removing an edge creates a valid tree
	canFormTree := func(skipEdge []int) bool {
		dsu := theory.NewDSU(n + 1)
		for _, edge := range edges {
			if edge[0] == skipEdge[0] && edge[1] == skipEdge[1] {
				continue
			}
			if dsu.SameComponent(edge[0], edge[1]) {
				return false
			}
			dsu.Union(edge[0], edge[1])
		}
		return true
	}

	// Case 1: No node with in-degree 2
	if node2 == 0 {
		dsu := theory.NewDSU(n + 1)
		for _, edge := range edges {
			if dsu.SameComponent(edge[0], edge[1]) {
				return edge
			}
			dsu.Union(edge[0], edge[1])
		}
		return nil
	}

	// Case 2: Node with in-degree 2 exists
	// Try removing the second incoming edge first
	incomingEdges := parent[node2]
	if canFormTree(incomingEdges[1]) {
		return incomingEdges[1]
	}
	// If there's still a cycle, remove the first incoming edge
	return incomingEdges[0]
}
