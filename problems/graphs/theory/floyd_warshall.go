package theory

import "math"

// Флойд-Уоршелл для всех пар кратчайших путей
func FloydWarshall(vertices int, edges [][]int) [][]int {
	// Инициализировать матрицу расстояний
	INF := math.MaxInt / 2
	dist := make([][]int, vertices)
	for i := range dist {
		dist[i] = make([]int, vertices)
		for j := range dist[i] {
			if i == j {
				dist[i][j] = 0
			} else {
				dist[i][j] = INF
			}
		}
	}

	// Добавить рёбра
	for _, edge := range edges {
		u, v, w := edge[0], edge[1], edge[2]
		dist[u][v] = w
	}

	// DP: для каждого промежуточного узла k
	for k := 0; k < vertices; k++ {
		for i := 0; i < vertices; i++ {
			for j := 0; j < vertices; j++ {
				if dist[i][k] != INF && dist[k][j] != INF {
					dist[i][j] = min(dist[i][j], dist[i][k]+dist[k][j])
				}
			}
		}
	}

	return dist
}

// Обнаружить отрицательный цикл
func HasNegativeCycle(vertices int, dist [][]int) bool {
	for i := 0; i < vertices; i++ {
		if dist[i][i] < 0 {
			return true
		}
	}
	return false
}

// Восстановить путь от i к j
func ReconstructPath(dist [][]int, next [][]int, i, j int) []int {
	if dist[i][j] == math.MaxInt/2 {
		return nil // Нет пути
	}

	path := []int{i}
	for i != j {
		i = next[i][j]
		path = append(path, i)
	}
	return path
}

// Флойд-Уоршелл с восстановлением пути
func FloydWarshallWithPath(vertices int, edges [][]int) ([][]int, [][]int) {
	INF := math.MaxInt / 2
	dist := make([][]int, vertices)
	next := make([][]int, vertices)

	for i := range dist {
		dist[i] = make([]int, vertices)
		next[i] = make([]int, vertices)
		for j := range dist[i] {
			if i == j {
				dist[i][j] = 0
				next[i][j] = i
			} else {
				dist[i][j] = INF
				next[i][j] = -1
			}
		}
	}

	// Добавить рёбра
	for _, edge := range edges {
		u, v, w := edge[0], edge[1], edge[2]
		dist[u][v] = w
		next[u][v] = v
	}

	// DP
	for k := 0; k < vertices; k++ {
		for i := 0; i < vertices; i++ {
			for j := 0; j < vertices; j++ {
				if dist[i][k] != INF && dist[k][j] != INF {
					if dist[i][k]+dist[k][j] < dist[i][j] {
						dist[i][j] = dist[i][k] + dist[k][j]
						next[i][j] = next[i][k]
					}
				}
			}
		}
	}

	return dist, next
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
