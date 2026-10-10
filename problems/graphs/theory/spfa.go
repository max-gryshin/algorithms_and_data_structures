package theory

import "math"

// SPFA — улучшенная версия Беллмана-Форда
func SPFA(vertices int, edges []Edge, source int) ([]int, bool) {
	distances := make([]int, vertices)
	inQueue := make([]bool, vertices)
	count := make([]int, vertices)

	for i := range distances {
		distances[i] = math.MaxInt / 2
	}
	distances[source] = 0

	queue := []int{source}
	inQueue[source] = true

	for len(queue) > 0 {
		u := queue[0]
		queue = queue[1:]
		inQueue[u] = false

		for _, edge := range edges {
			if edge.u != u {
				continue
			}

			v := edge.v
			if distances[u] != math.MaxInt/2 &&
				distances[u]+edge.weight < distances[v] {
				distances[v] = distances[u] + edge.weight
				count[v]++

				// Обнаружить отрицательный цикл
				if count[v] >= vertices {
					return nil, true
				}

				if !inQueue[v] {
					queue = append(queue, v)
					inQueue[v] = true
				}
			}
		}
	}

	return distances, false
}
