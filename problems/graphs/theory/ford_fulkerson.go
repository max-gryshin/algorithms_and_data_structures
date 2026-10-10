package theory

import "math"

// Ford-Fulkerson алгоритм
func FordFulkerson(graph [][]int, source, sink int) int {
	n := len(graph)
	capacity := make([][]int, n)
	for i := range capacity {
		capacity[i] = make([]int, n)
		copy(capacity[i], graph[i])
	}

	maxFlow := 0

	for {
		// Найти увеличивающий путь из source в sink
		parent := make([]int, n)
		for i := range parent {
			parent[i] = -1
		}

		// BFS для нахождения пути
		queue := []int{source}
		parent[source] = source

		for len(queue) > 0 && parent[sink] == -1 {
			u := queue[0]
			queue = queue[1:]

			for v := 0; v < n; v++ {
				if parent[v] == -1 && capacity[u][v] > 0 {
					parent[v] = u
					queue = append(queue, v)

					if v == sink {
						break
					}
				}
			}
		}

		if parent[sink] == -1 {
			break // Нет пути
		}

		// Найти минимальную пропускную способность на пути
		pathFlow := math.MaxInt
		for v := sink; v != source; v = parent[v] {
			u := parent[v]
			pathFlow = min(pathFlow, capacity[u][v])
		}

		// Обновить пропускные способности
		for v := sink; v != source; v = parent[v] {
			u := parent[v]
			capacity[u][v] -= pathFlow
			capacity[v][u] += pathFlow // Обратный поток
		}

		maxFlow += pathFlow
	}

	return maxFlow
}
