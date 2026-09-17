package dfs_bfs

func kahnTopologicalSort(graph map[int][]int) []int {
	indegree := make(map[int]int)
	for vertex := range graph {
		indegree[vertex] = 0
	}
	for _, neighbors := range graph {
		for _, neighbor := range neighbors {
			indegree[neighbor]++
		}
	}
	queue := []int{}
	for vertex, degree := range indegree {
		if degree == 0 {
			queue = append(queue, vertex)
		}
	}

	res := make([]int, 0)
	for len(queue) > 0 {
		// pop from queue
		vertex := queue[0]
		queue = queue[1:]
		// since queue contains only vertex with the 0 indegree
		// add it to result
		res = append(res, vertex)
		for _, neighbor := range graph[vertex] {
			// decrease the amout of degree of neighbors of vertex
			indegree[neighbor]--
			if indegree[neighbor] == 0 {
				queue = append(queue, neighbor)
			}
		}
	}

	return res
}
