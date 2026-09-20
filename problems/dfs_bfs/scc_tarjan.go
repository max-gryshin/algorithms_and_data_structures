package dfs_bfs

type TarjanSCC struct {
	n int
	// adjacency list (outdegrees)
	adj [][]int
	// counter
	timer   int
	ids     []int
	low     []int
	inStack []bool
	stack   []int
	sccs    [][]int
}

func NewTarjanSCC(n int, edges [][]int) *TarjanSCC {
	adj := make([][]int, n+1)
	for _, edge := range edges {
		u, v := edge[0], edge[1]
		adj[u] = append(adj[u], v)
	}

	return &TarjanSCC{
		n:       n,
		adj:     adj,
		timer:   0,
		ids:     make([]int, n+1),
		low:     make([]int, n+1),
		inStack: make([]bool, n+1),
		stack:   make([]int, 0),
		sccs:    make([][]int, 0),
	}
}

func (t *TarjanSCC) dfs(u int) {
	t.timer++
	t.ids[u] = t.timer
	t.low[u] = t.timer
	t.stack = append(t.stack, u)
	t.inStack[u] = true

	// travserse by all vertices of the current node
	// and deep dive by dfs
	for _, v := range t.adj[u] {
		// vertex not visited yet
		if t.ids[v] == 0 {
			t.dfs(v)
			t.low[u] = min(t.low[u], t.low[v])
		} else if t.inStack[v] { // vertex v already in stack
			t.low[u] = min(t.low[u], t.ids[v])
		}
	}

	// if u a root of SCC
	if t.low[u] == t.ids[u] {
		var currentSCC []int
		for {
			// pop from stack
			top := t.stack[len(t.stack)-1]
			t.stack = t.stack[:len(t.stack)-1]
			t.inStack[top] = false

			currentSCC = append(currentSCC, top)
			if top == u {
				break
			}
		}
		t.sccs = append(t.sccs, currentSCC)
	}
}

func (t *TarjanSCC) FindSCC() [][]int {
	// for untouched vertices
	for i := 1; i <= t.n; i++ {
		if t.ids[i] == 0 {
			t.dfs(i)
		}
	}
	return t.sccs
}
