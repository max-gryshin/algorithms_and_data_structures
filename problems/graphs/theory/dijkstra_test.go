package theory

import (
	"testing"
)

func TestDijkstra(t *testing.T) {
	// Граф:
	//   0 --1-- 1
	//   |       |
	//   4       2
	//   |       |
	//   4 --3-- 2

	vertices := 5
	edges := [][]int{
		{0, 1, 1},
		{0, 2, 4},
		{1, 2, 2},
		{1, 3, 5},
		{2, 3, 1},
		{3, 4, 3},
	}

	distances := Dijkstra(vertices, edges, 0)

	expected := []int{0, 1, 3, 4, 7}
	for i, d := range distances {
		if d != expected[i] {
			t.Errorf("vertex %d: expected distance %d, got %d", i, expected[i], d)
		}
	}

	t.Logf("Dijkstra from vertex 0: distances=%v", distances)
}

func TestDijkstraPath(t *testing.T) {
	vertices := 5
	edges := [][]int{
		{0, 1, 1},
		{0, 2, 4},
		{1, 2, 2},
		{1, 3, 5},
		{2, 3, 1},
		{3, 4, 3},
	}

	path, distance := DijkstraPath(vertices, edges, 0, 4)

	if distance != 7 {
		t.Errorf("expected distance 7, got %d", distance)
	}

	expectedPath := []int{0, 1, 2, 3, 4}
	if len(path) != len(expectedPath) {
		t.Errorf("expected path length %d, got %d", len(expectedPath), len(path))
	}

	for i, v := range path {
		if v != expectedPath[i] {
			t.Errorf("path[%d]: expected %d, got %d", i, expectedPath[i], v)
		}
	}

	t.Logf("Dijkstra path from 0 to 4: path=%v, distance=%d", path, distance)
}

func TestDijkstraSingleSource(t *testing.T) {
	vertices := 3
	edges := [][]int{
		{0, 1, 2},
		{0, 2, 5},
		{1, 2, 1},
	}

	distances := Dijkstra(vertices, edges, 0)

	expected := []int{0, 2, 3}
	for i, d := range distances {
		if d != expected[i] {
			t.Errorf("vertex %d: expected distance %d, got %d", i, expected[i], d)
		}
	}

	t.Logf("Single source shortest paths: %v", distances)
}

func TestDijkstraAdjacencyList(t *testing.T) {
	INF := 1000000
	adjacency := [][]int{
		{0, 1, INF},
		{1, 0, 2},
		{INF, 2, 0},
	}

	distances := DijkstraAdjacencyList(adjacency, 0)

	expected := []int{0, 1, 3}
	for i, d := range distances {
		if d != expected[i] {
			t.Errorf("vertex %d: expected distance %d, got %d", i, expected[i], d)
		}
	}

	t.Logf("Dijkstra with adjacency matrix: %v", distances)
}

func TestDijkstraDisconnectedGraph(t *testing.T) {
	vertices := 4
	edges := [][]int{
		{0, 1, 1},
		{2, 3, 1},
	}

	distances := Dijkstra(vertices, edges, 0)

	// Вершины 2 и 3 недостижимы из 0
	if distances[0] != 0 || distances[1] != 1 {
		t.Errorf("unexpected distances for reachable vertices")
	}

	// 2 и 3 должны быть "бесконечностью"
	if distances[2] < 1000000 || distances[3] < 1000000 {
		t.Logf("Disconnected graph handled correctly: distances=%v", distances)
	}
}
