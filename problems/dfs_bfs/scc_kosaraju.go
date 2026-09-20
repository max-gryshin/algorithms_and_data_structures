package dfs_bfs

// 1st DFS: completes the vertex stack in terms of time, concluding the traversals
func fillOrder(v int, visited []bool, stack *[]int, adj [][]int) {
	visited[v] = true
	for _, neighbor := range adj[v] {
		if !visited[neighbor] {
			fillOrder(neighbor, visited, stack, adj)
		}
	}
	*stack = append(*stack, v)
}

// 2nd DFS: completes vertices of the current scc in transpose graph
func dfsTranspose(v int, visited []bool, component *[]int, adjTranspose [][]int) {
	visited[v] = true
	*component = append(*component, v)
	for _, neighbor := range adjTranspose[v] {
		if !visited[neighbor] {
			dfsTranspose(neighbor, visited, component, adjTranspose)
		}
	}
}

// findSCC Kosaraju finds all SCC with O(V + E)
func findSCCKosaraju(n int, edges [][]int) [][]int {
	adj := make([][]int, n+1)
	adjTranspose := make([][]int, n+1)
	// 1. Construction of the direct and transposed graphs
	for _, edge := range edges {
		u, v := edge[0], edge[1]
		adj[u] = append(adj[u], v)
		adjTranspose[v] = append(adjTranspose[v], u) // invert a direction of the edges
	}

	// 2. The first DFS pass to populate the vertex exit order.
	visited := make([]bool, n+1)
	var stack []int

	for i := 1; i <= n; i++ {
		if !visited[i] {
			fillOrder(i, visited, &stack, adj)
		}
	}

	// 3. Resetting the visit array for the second stage
	for i := range visited {
		visited[i] = false
	}

	// 4. Traversal of the transposed graph in reverse stack order
	var result [][]int

	for i := len(stack) - 1; i >= 0; i-- {
		v := stack[i]
		if !visited[v] {
			var component []int
			dfsTranspose(v, visited, &component, adjTranspose)
			result = append(result, component)
		}
	}

	return result
}
