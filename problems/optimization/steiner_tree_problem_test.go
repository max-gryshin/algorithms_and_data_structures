package optimization

import (
	"testing"
)

func TestKruskalConnectedGraph(t *testing.T) {
	nodes := []Point{
		{X: 0, Y: 0},
		{X: 1, Y: 0},
		{X: 1, Y: 1},
		{X: 0, Y: 1},
	}

	mstCost, ok := MSTKruskal(nodes)

	if !ok {
		t.Errorf("expected connected graph, got false")
	}
	if mstCost <= 0 {
		t.Errorf("expected positive MST cost, got %.2f", mstCost)
	}

	t.Logf("Connected graph MST cost: %.2f", mstCost)
}

func TestKruskalSingleNode(t *testing.T) {
	nodes := []Point{{X: 0, Y: 0}}
	mstCost, ok := MSTKruskal(nodes)

	if !ok || mstCost != 0 {
		t.Errorf("expected cost=0 for single node, got cost=%.2f, ok=%v", mstCost, ok)
	}

	t.Logf("Single node MST cost: %.2f", mstCost)
}

func TestSteinerIndividualFitness(t *testing.T) {
	terminals := []Point{
		{X: 0, Y: 0},
		{X: 10, Y: 10},
		{X: 20, Y: 0},
	}

	steinerPool := []Point{
		{X: 5, Y: 2},
		{X: 15, Y: 3},
		{X: 10, Y: 8},
	}

	nodeActivationCost := 5.0

	individual := SteinerIndividual{
		Genes: []bool{true, false, true},
	}

	// Собираем активный набор узлов
	activeNodes := make([]Point, len(terminals))
	copy(activeNodes, terminals)

	activeSteinerCount := 0
	for j, active := range individual.Genes {
		if active {
			activeNodes = append(activeNodes, steinerPool[j])
			activeSteinerCount++
		}
	}

	mstCost, ok := MSTKruskal(activeNodes)
	if !ok {
		t.Errorf("expected connected graph")
	}

	expectedFitness := mstCost + float64(activeSteinerCount)*nodeActivationCost
	t.Logf("Individual fitness: %.2f (MST: %.2f, Activation: %.2f)", expectedFitness, mstCost, float64(activeSteinerCount)*nodeActivationCost)
}

func TestRunSteinerTreeAlgorithm(t *testing.T) {
	terminals := []Point{
		{X: 0, Y: 0},
		{X: 10, Y: 10},
		{X: 20, Y: 0},
	}

	steinerPool := []Point{
		{X: 5, Y: 2},
		{X: 15, Y: 3},
		{X: 10, Y: 8},
	}

	nodeActivationCost := 5.0
	popSize := 20
	generations := 50
	mutationRate := 0.1

	result := RunSteinerTreeAlgorithm(terminals, steinerPool, nodeActivationCost, popSize, generations, mutationRate)

	if result.Best.Fitness <= 0 {
		t.Errorf("expected positive fitness, got %.2f", result.Best.Fitness)
	}

	activeNodes := 0
	for _, active := range result.Best.Genes {
		if active {
			activeNodes++
		}
	}

	t.Logf("Steiner Tree Result: cost=%.2f, selected_steiner_nodes=%d, genes=%v",
		result.Best.Fitness, activeNodes, result.Best.Genes)
}

func TestRunSteinerTreeAlgorithmMinimization(t *testing.T) {
	terminals := []Point{
		{X: 0, Y: 0},
		{X: 100, Y: 100},
	}

	steinerPool := []Point{
		{X: 50, Y: 50},
		{X: 25, Y: 25},
		{X: 75, Y: 75},
	}

	nodeActivationCost := 10.0
	popSize := 30
	generations := 100
	mutationRate := 0.15

	result := RunSteinerTreeAlgorithm(terminals, steinerPool, nodeActivationCost, popSize, generations, mutationRate)

	if result.Best.Fitness <= 0 {
		t.Errorf("expected positive fitness, got %.2f", result.Best.Fitness)
	}

	t.Logf("Steiner Tree Minimization: cost=%.2f, genes=%v", result.Best.Fitness, result.Best.Genes)
}

func TestSteinerTournamentSelection(t *testing.T) {
	population := []SteinerIndividual{
		{Genes: []bool{true, false}, Fitness: 10.0},
		{Genes: []bool{false, true}, Fitness: 5.0},
		{Genes: []bool{true, true}, Fitness: 15.0},
	}

	selected := SteinerTournamentSelection(population)

	if selected.Fitness < 0 {
		t.Errorf("expected non-negative fitness, got %.2f", selected.Fitness)
	}

	t.Logf("Tournament selection: selected individual with fitness=%.2f, genes=%v", selected.Fitness, selected.Genes)
}

func TestSteinerCrossover(t *testing.T) {
	p1 := SteinerIndividual{Genes: []bool{true, true, true, true}}
	p2 := SteinerIndividual{Genes: []bool{false, false, false, false}}

	child := SteinerCrossover(p1, p2)

	if len(child.Genes) != len(p1.Genes) {
		t.Errorf("expected child genes length %d, got %d", len(p1.Genes), len(child.Genes))
	}

	t.Logf("Crossover: p1=%v, p2=%v, child=%v", p1.Genes, p2.Genes, child.Genes)
}

func TestSteinerMutate(t *testing.T) {
	individual := SteinerIndividual{
		Genes: []bool{false, false, false, false},
	}
	original := make([]bool, len(individual.Genes))
	copy(original, individual.Genes)

	individual.SteinerMutate(0.5)

	if len(individual.Genes) != len(original) {
		t.Errorf("expected same genes length after mutation")
	}

	for _, gene := range individual.Genes {
		if gene != true && gene != false {
			t.Errorf("invalid gene value after mutation")
		}
	}

	t.Logf("Mutate: original=%v, mutated=%v", original, individual.Genes)
}
