package theory

import (
	"container/heap"
	"math"
)

// Dijkstra находит кратчайшие пути от исходной вершины до всех остальных
// Используется приоритетная очередь и работает только с неотрицательными весами
func Dijkstra(vertices int, edges [][]int, source int) []int {
	distances := make([]int, vertices)
	for i := range distances {
		distances[i] = math.MaxInt / 2 // "бесконечность"
	}
	distances[source] = 0

	// Приоритетная очередь: (расстояние, вершина)
	pq := &PriorityQueue{}
	heap.Init(pq)
	heap.Push(pq, &Item{vertex: source, distance: 0})

	visited := make([]bool, vertices)

	for pq.Len() > 0 {
		item := heap.Pop(pq).(*Item)
		vertex := item.vertex

		if visited[vertex] {
			continue
		}
		visited[vertex] = true

		// Релаксация соседних вершин
		for _, edge := range edges {
			u, v, w := edge[0], edge[1], edge[2]

			// Если это ребро от текущей вершины
			if u == vertex && !visited[v] {
				if distances[u]+w < distances[v] {
					distances[v] = distances[u] + w
					heap.Push(pq, &Item{vertex: v, distance: distances[v]})
				}
			}
		}
	}

	return distances
}

// DijkstraPath находит кратчайший путь от source до target
func DijkstraPath(vertices int, edges [][]int, source, target int) ([]int, int) {
	distances := make([]int, vertices)
	parent := make([]int, vertices)

	for i := range distances {
		distances[i] = math.MaxInt / 2
		parent[i] = -1
	}
	distances[source] = 0

	pq := &PriorityQueue{}
	heap.Init(pq)
	heap.Push(pq, &Item{vertex: source, distance: 0})

	visited := make([]bool, vertices)

	for pq.Len() > 0 {
		item := heap.Pop(pq).(*Item)
		vertex := item.vertex

		if visited[vertex] {
			continue
		}
		visited[vertex] = true

		for _, edge := range edges {
			u, v, w := edge[0], edge[1], edge[2]

			if u == vertex && !visited[v] {
				if distances[u]+w < distances[v] {
					distances[v] = distances[u] + w
					parent[v] = u
					heap.Push(pq, &Item{vertex: v, distance: distances[v]})
				}
			}
		}
	}

	// Восстановить путь от source до target
	path := []int{}
	for v := target; v != -1; v = parent[v] {
		path = append([]int{v}, path...)
	}

	return path, distances[target]
}

// DijkstraAdjacencyList - версия Дейкстры с матрицей смежности
// Полезна когда граф задан как матрица весов
func DijkstraAdjacencyList(adjacency [][]int, source int) []int {
	vertices := len(adjacency)
	distances := make([]int, vertices)
	visited := make([]bool, vertices)

	for i := range distances {
		distances[i] = math.MaxInt / 2
	}
	distances[source] = 0

	for i := 0; i < vertices; i++ {
		// Найти непосещённую вершину с минимальным расстоянием
		minDist := math.MaxInt / 2
		minVertex := -1

		for v := 0; v < vertices; v++ {
			if !visited[v] && distances[v] < minDist {
				minDist = distances[v]
				minVertex = v
			}
		}

		if minVertex == -1 {
			break
		}

		visited[minVertex] = true

		// Релаксация соседей
		for v := 0; v < vertices; v++ {
			if adjacency[minVertex][v] != math.MaxInt/2 && !visited[v] {
				if distances[minVertex]+adjacency[minVertex][v] < distances[v] {
					distances[v] = distances[minVertex] + adjacency[minVertex][v]
				}
			}
		}
	}

	return distances
}

type Item struct {
	vertex   int
	distance int
	index    int
}

type PriorityQueue []*Item

func (pq PriorityQueue) Len() int {
	return len(pq)
}

func (pq PriorityQueue) Less(i, j int) bool {
	return pq[i].distance < pq[j].distance
}

func (pq PriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

func (pq *PriorityQueue) Push(x interface{}) {
	n := len(*pq)
	item := x.(*Item)
	item.index = n
	*pq = append(*pq, item)
}

func (pq *PriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	old[n-1] = nil
	item.index = -1
	*pq = old[0 : n-1]
	return item
}
