package optimization

import (
	"testing"
)

func TestCityDistanceTo(t *testing.T) {
	c1 := City{X: 0, Y: 0}
	c2 := City{X: 3, Y: 4}
	dist := c1.DistanceTo(c2)
	expected := 5.0
	if dist != expected {
		t.Errorf("expected distance %.2f, got %.2f", expected, dist)
	}
	t.Logf("Distance between (0,0) and (3,4): %.2f", dist)
}

func TestInitCities(t *testing.T) {
	initCities()
	if len(cities) != numCities {
		t.Errorf("expected %d cities, got %d", numCities, len(cities))
	}
	for i, city := range cities {
		if city.X < 0 || city.Y < 0 {
			t.Errorf("city %d has invalid coordinates: (%.2f, %.2f)", i, city.X, city.Y)
		}
	}
	t.Logf("Initialized %d cities", len(cities))
}

func TestTSPIndividualEvaluate(t *testing.T) {
	initCities()
	route := make([]int, numCities)
	for i := 0; i < numCities; i++ {
		route[i] = i
	}
	ind := TSPIndividual{Route: route}
	ind.Evaluate()

	if ind.Fitness <= 0 {
		t.Errorf("expected positive fitness, got %.4f", ind.Fitness)
	}
	totalDist := 1.0 / ind.Fitness
	t.Logf("Route: %v, Total distance: %.2f, Fitness: %.6f", route, totalDist, ind.Fitness)
}

func TestOrderedCrossover(t *testing.T) {
	parent1 := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	parent2 := []int{9, 8, 7, 6, 5, 4, 3, 2, 1, 0}

	child := OrderedCrossover(parent1, parent2)

	if len(child) != len(parent1) {
		t.Errorf("expected child length %d, got %d", len(parent1), len(child))
	}

	seen := make(map[int]bool)
	for _, city := range child {
		if city < 0 || city >= numCities {
			t.Errorf("invalid city index in child: %d", city)
		}
		if seen[city] {
			t.Errorf("duplicate city in child: %d", city)
		}
		seen[city] = true
	}
	t.Logf("Crossover: parent1=%v, parent2=%v, child=%v", parent1, parent2, child)
}

func TestTSPIndividualMutate(t *testing.T) {
	route := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	ind := TSPIndividual{Route: route}
	original := make([]int, len(route))
	copy(original, ind.Route)

	ind.Mutate()

	if len(ind.Route) != len(original) {
		t.Errorf("expected route length %d, got %d", len(original), len(ind.Route))
	}

	seen := make(map[int]bool)
	for _, city := range ind.Route {
		if city < 0 || city >= numCities {
			t.Errorf("invalid city index: %d", city)
		}
		seen[city] = true
	}

	if len(seen) != numCities {
		t.Errorf("expected %d unique cities, got %d", numCities, len(seen))
	}
	t.Logf("Mutate: original=%v, mutated=%v", original, ind.Route)
}

func TestTSPTournamentSelection(t *testing.T) {
	population := []TSPIndividual{
		{Route: []int{0, 1, 2}, Fitness: 0.1},
		{Route: []int{2, 1, 0}, Fitness: 0.05},
		{Route: []int{1, 0, 2}, Fitness: 0.08},
	}

	selected := TSPTournamentSelection(population)
	if selected.Fitness <= 0 {
		t.Errorf("expected positive fitness, got %.4f", selected.Fitness)
	}
	t.Logf("Tournament selection: selected route=%v with fitness=%.6f", selected.Route, selected.Fitness)
}

func TestRunTSPAlgorithm(t *testing.T) {
	result := RunTSPAlgorithm()

	if len(result.Route) != numCities {
		t.Errorf("expected route length %d, got %d", numCities, len(result.Route))
	}

	if result.Fitness <= 0 {
		t.Errorf("expected positive fitness, got %.4f", result.Fitness)
	}

	seen := make(map[int]bool)
	for _, city := range result.Route {
		if city < 0 || city >= numCities {
			t.Errorf("invalid city index: %d", city)
		}
		seen[city] = true
	}

	if len(seen) != numCities {
		t.Errorf("expected %d unique cities, got %d", numCities, len(seen))
	}

	totalDist := 1.0 / result.Fitness
	t.Logf("TSP Result: route=%v, total distance=%.2f, fitness=%.6f", result.Route, totalDist, result.Fitness)
}
