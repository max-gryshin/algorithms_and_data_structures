package theory

// Disjoint Set Union
type DSU struct {
	// set (tree) where indexes are vertices (nodes)
	// and values represents a root
	Parent []int
	// Rank is an upper bound on the tree height,
	// used for union by Rank optimization to keep trees balanced
	Rank []int
}

func NewDSU(graphSize int) *DSU {
	parent := make([]int, graphSize)
	rank := make([]int, graphSize)
	// make set
	for i := range parent {
		parent[i] = i
	}

	return &DSU{
		Parent: parent,
		Rank:   rank,
	}
}

// BuildComponents processes the graph to identify connected components using the union-find algorithm.
func (d *DSU) BuildComponents(graph [][]int) {
	for _, edge := range graph {
		x, y := edge[0], edge[1]
		// if roots of the x and y are not equals
		// then union the
		if d.FindSet(x) != d.FindSet(y) {
			d.Union(x, y)
		}
	}
}

// FindSet returns the representative (root) of the set containing the element x, with path compression for optimization.
// If x is not its own parent in the subset tree, FindSet recursively updates and flattens the path to the root.
func (d *DSU) FindSet(x int) int {
	// if x is not a root
	if d.Parent[x] != x {
		// smooth the tree by redirecting the node toward the root.
		d.Parent[x] = d.FindSet(d.Parent[x]) // path compression
	}

	return d.Parent[x]
}

// Union merges the sets containing elements x and y into a single set by linking their roots using union by rank.
func (d *DSU) Union(x, y int) {
	d.link(d.FindSet(x), d.FindSet(y))
}

// link merges two subsets by linking their roots using the union by rank heuristic to maintain balanced trees.
func (d *DSU) link(x, y int) {
	// union by rank
	if d.Rank[x] < d.Rank[y] {
		d.Parent[x] = y
	} else if d.Rank[x] > d.Rank[y] {
		d.Parent[y] = x
	} else {
		d.Parent[y] = x
		d.Rank[x]++
	}
}

// SameComponent checks whether the elements x and y belong to the same connected component.
func (d *DSU) SameComponent(x, y int) bool {
	return d.FindSet(x) == d.FindSet(y)
}
