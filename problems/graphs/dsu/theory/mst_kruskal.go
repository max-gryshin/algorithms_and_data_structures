package theory

import (
	"sort"
)

// Edge represents a graph edge
type Edge struct {
	Source, Destination int
	Weight              int
}

// Graph represents a weighted undirected graph
type Graph struct {
	Vertices int
	Edges    []Edge
}

// MST - Minimum spanning tree
// MSTKruskal finds the minimum spanning tree of the graph
func MSTKruskal(g Graph) []Edge {
	// A = ∅ (initialize empty set of MST edges)
	var mst []Edge

	// Initialize disjoint set structure
	// for each vertex v ∈ G.V: MAKE-SET(v)
	dsu := NewDSU(g.Vertices)

	// sort the edges of G.E into nondecreasing order by weight w
	// Make a copy to avoid mutating the original graph
	sortedEdges := make([]Edge, len(g.Edges))
	copy(sortedEdges, g.Edges)
	sort.Slice(sortedEdges, func(i, j int) bool {
		return sortedEdges[i].Weight < sortedEdges[j].Weight
	})

	// for each edge (u, v) ∈ G.E, taken in nondecreasing order by weight:
	for _, edge := range sortedEdges {
		// if FIND-SET(u) ≠ FIND-SET(v)
		if dsu.FindSet(edge.Source) != dsu.FindSet(edge.Destination) {
			// A = A ∪ {(u, v)}
			mst = append(mst, edge)
			// UNION(u, v)
			dsu.Union(edge.Source, edge.Destination)
		}
	}

	// return A
	return mst
}
