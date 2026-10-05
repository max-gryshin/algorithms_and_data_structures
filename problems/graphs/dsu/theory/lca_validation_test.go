package theory

import (
	"slices"
	"testing"
)

func TestLCAValidation(t *testing.T) {
	tests := []struct {
		name            string
		numVertices     int
		edges           [][2]int // parent, child pairs
		queries         []Query
		expectedAnswers []int
	}{
		{
			name:        "Simple linear tree",
			numVertices: 4,
			edges:       [][2]int{{0, 1}, {1, 2}, {2, 3}},
			queries: []Query{
				{Index: 0, U: 0, V: 3}, // LCA(0, 3) = 0
				{Index: 1, U: 1, V: 3}, // LCA(1, 3) = 1
				{Index: 2, U: 2, V: 3}, // LCA(2, 3) = 2
			},
			expectedAnswers: []int{0, 1, 2},
		},
		{
			name:        "Complete binary tree",
			numVertices: 7,
			edges: [][2]int{
				{0, 1}, {0, 2},
				{1, 3}, {1, 4},
				{2, 5}, {2, 6},
			},
			queries: []Query{
				{Index: 0, U: 3, V: 4}, // LCA(3, 4) = 1 (siblings)
				{Index: 1, U: 3, V: 5}, // LCA(3, 5) = 0
				{Index: 2, U: 5, V: 6}, // LCA(5, 6) = 2 (siblings)
				{Index: 3, U: 1, V: 2}, // LCA(1, 2) = 0 (siblings)
			},
			expectedAnswers: []int{1, 0, 2, 0},
		},
		{
			name:        "Unbalanced tree",
			numVertices: 5,
			edges: [][2]int{
				{0, 1},
				{1, 2},
				{0, 3},
				{3, 4},
			},
			queries: []Query{
				{Index: 0, U: 2, V: 4}, // LCA(2, 4) = 0
				{Index: 1, U: 1, V: 4}, // LCA(1, 4) = 0
				{Index: 2, U: 2, V: 1}, // LCA(2, 1) = 1
				{Index: 3, U: 3, V: 4}, // LCA(3, 4) = 3
			},
			expectedAnswers: []int{0, 0, 1, 3},
		},
		{
			name:        "Same node queries",
			numVertices: 4,
			edges:       [][2]int{{0, 1}, {1, 2}, {1, 3}},
			queries: []Query{
				{Index: 0, U: 0, V: 0}, // LCA(0, 0) = 0
				{Index: 1, U: 2, V: 2}, // LCA(2, 2) = 2
				{Index: 2, U: 0, V: 1}, // LCA(0, 1) = 0
			},
			expectedAnswers: []int{0, 2, 0},
		},
		{
			name:        "Star graph",
			numVertices: 6,
			edges:       [][2]int{{0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}},
			queries: []Query{
				{Index: 0, U: 1, V: 2}, // LCA(1, 2) = 0
				{Index: 1, U: 3, V: 5}, // LCA(3, 5) = 0
				{Index: 2, U: 1, V: 4}, // LCA(1, 4) = 0
				{Index: 3, U: 2, V: 3}, // LCA(2, 3) = 0
			},
			expectedAnswers: []int{0, 0, 0, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tree := buildTreeFromEdges(tt.numVertices, tt.edges, tt.queries)
			dsu := NewDisjointSetUnionLCA(tt.numVertices)
			answers := make([]int, len(tt.queries))

			dsu.LCA(0, tree, answers)

			if !slices.Equal(answers, tt.expectedAnswers) {
				t.Errorf("expected %v, got %v", tt.expectedAnswers, answers)
			}
		})
	}
}

func buildTreeFromEdges(numVertices int, edges [][2]int, queries []Query) []Node {
	tree := make([]Node, numVertices)
	for i := 0; i < numVertices; i++ {
		tree[i] = Node{ID: i}
	}

	// Build tree from edges (parent, child pairs)
	for _, edge := range edges {
		parent, child := edge[0], edge[1]
		tree[parent].Children = append(tree[parent].Children, child)
	}

	// Distribute queries to both endpoints
	for _, query := range queries {
		tree[query.U].Queries = append(tree[query.U].Queries, query)
		tree[query.V].Queries = append(tree[query.V].Queries, query)
	}

	return tree
}

// Verify LCA correctness using naive algorithm for comparison
func TestLCACorrectnessVsNaive(t *testing.T) {
	// Build a test tree
	edges := [][2]int{
		{0, 1}, {0, 2},
		{1, 3}, {1, 4},
		{2, 5}, {2, 6},
		{6, 7},
	}
	numVertices := 8

	// Generate all possible queries
	var queries []Query
	idx := 0
	for u := 0; u < numVertices; u++ {
		for v := u; v < numVertices; v++ {
			queries = append(queries, Query{Index: idx, U: u, V: v})
			idx++
		}
	}

	// Build tree
	tree := buildTreeFromEdges(numVertices, edges, queries)

	// Run Tarjan algorithm
	dsu := NewDisjointSetUnionLCA(numVertices)
	answers := make([]int, len(queries))
	dsu.LCA(0, tree, answers)

	// Verify with naive algorithm
	parent := make([]int, numVertices)
	parent[0] = -1
	buildParentArray(tree, 0, -1, parent)

	for i, query := range queries {
		expected := naiveLCA(query.U, query.V, parent)
		if answers[i] != expected {
			t.Errorf("Query(%d, %d): expected %d, got %d", query.U, query.V, expected, answers[i])
		}
	}
}

func buildParentArray(tree []Node, u, p int, parent []int) {
	parent[u] = p
	for _, v := range tree[u].Children {
		buildParentArray(tree, v, u, parent)
	}
}

func naiveLCA(u, v int, parent []int) int {
	// Find all ancestors of u
	ancestors := make(map[int]bool)
	curr := u
	for curr != -1 {
		ancestors[curr] = true
		curr = parent[curr]
	}

	// Find first common ancestor by traversing up from v
	curr = v
	for curr != -1 {
		if ancestors[curr] {
			return curr
		}
		curr = parent[curr]
	}
	return -1
}
