package theory

import "math"

type Edge struct {
	u, v, weight int
}

// Беллман-Форд для нахождения кратчайших путей
func BellmanFord(vertices int, edges []Edge, source int) ([]int, bool) {
	distances := make([]int, vertices)
	for i := range distances {
		distances[i] = math.MaxInt / 2 // "бесконечность"
	}
	distances[source] = 0

	// V-1 релаксаций
	for i := 0; i < vertices-1; i++ {
		for _, edge := range edges {
			if distances[edge.u] != math.MaxInt/2 &&
				distances[edge.u]+edge.weight < distances[edge.v] {
				distances[edge.v] = distances[edge.u] + edge.weight
			}
		}
	}

	// Проверка отрицательного цикла
	for _, edge := range edges {
		if distances[edge.u] != math.MaxInt/2 &&
			distances[edge.u]+edge.weight < distances[edge.v] {
			return nil, true // Отрицательный цикл обнаружен
		}
	}

	return distances, false
}

// Восстановить путь от source до target
func BellmanFordPath(vertices int, edges []Edge, source, target int) ([]int, bool) {
	distances := make([]int, vertices)
	parent := make([]int, vertices)

	for i := range distances {
		distances[i] = math.MaxInt / 2
		parent[i] = -1
	}
	distances[source] = 0

	// V-1 релаксаций
	for i := 0; i < vertices-1; i++ {
		for _, edge := range edges {
			if distances[edge.u] != math.MaxInt/2 &&
				distances[edge.u]+edge.weight < distances[edge.v] {
				distances[edge.v] = distances[edge.u] + edge.weight
				parent[edge.v] = edge.u
			}
		}
	}

	// Проверка отрицательного цикла
	for _, edge := range edges {
		if distances[edge.u] != math.MaxInt/2 &&
			distances[edge.u]+edge.weight < distances[edge.v] {
			return nil, true
		}
	}

	// Восстановить путь
	path := []int{}
	for v := target; v != -1; v = parent[v] {
		path = append([]int{v}, path...)
	}

	return path, false
}
