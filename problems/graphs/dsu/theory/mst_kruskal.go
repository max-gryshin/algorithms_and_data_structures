package theory

import (
	"sort"
)

// Edge представляет ребро графа
type Edge struct {
	Source, Destination int
	Weight              int
}

// Graph представляет взвешенный неориентированный граф
type Graph struct {
	Vertices int
	Edges    []Edge
}

// MSTKruskal находит минимальное остовное дерево графа
func MSTKruskal(g Graph) []Edge {
	// A = ∅ (инициализация пустого множества ребер MST)
	var mst []Edge

	// Инициализируем структуру непересекающихся множеств
	// for each vertex v ∈ G.V: MAKE-SET(v)
	dsu := NewDSU(g.Vertices)

	// sort the edges of G.E into nondecreasing order by weight w
	// Делаем копию, чтобы не мутировать исходный граф
	sortedEdges := make([]Edge, len(g.Edges))
	copy(sortedEdges, g.Edges)
	sort.Slice(sortedEdges, func(i, j int) bool {
		return sortedEdges[i].Weight < sortedEdges[j].Weight
	})

	// for each edge (u, v) ∈ G.E, taken in nondecreasing order by weight:
	for _, edge := range sortedEdges {
		// if FIND-SET(u) ≠ FIND-SET(v)
		if dsu.Find(edge.Source) != dsu.Find(edge.Destination) {
			// A = A ∪ {(u, v)}
			mst = append(mst, edge)
			// UNION(u, v)
			dsu.Union(edge.Source, edge.Destination)
		}
	}

	// return A
	return mst
}
