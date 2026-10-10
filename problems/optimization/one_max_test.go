package optimization

import (
	"testing"
)

func TestGenerateRandomIndividual(t *testing.T) {
	ind := generateRandomIndividual(10)
	if len(ind.Genes) != 10 {
		t.Errorf("expected genes length 10, got %d", len(ind.Genes))
	}
	for _, gene := range ind.Genes {
		if gene != 0 && gene != 1 {
			t.Errorf("expected gene to be 0 or 1, got %d", gene)
		}
	}
	t.Logf("Generated individual with genes: %v", ind.Genes)
}

func TestEvaluate(t *testing.T) {
	ind := Individual{Genes: []int{1, 1, 0, 1, 1}}
	ind.Evaluate()
	if ind.Fitness != 4 {
		t.Errorf("expected fitness 4, got %d", ind.Fitness)
	}
	t.Logf("Evaluated individual: genes=%v, fitness=%d", ind.Genes, ind.Fitness)
}

func TestCrossover(t *testing.T) {
	parent1 := Individual{Genes: []int{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}
	parent2 := Individual{Genes: []int{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}

	child := Crossover(parent1, parent2)
	if len(child.Genes) != targetLen {
		t.Errorf("expected child genes length %d, got %d", targetLen, len(child.Genes))
	}
	t.Logf("Crossover: parent1=%v, parent2=%v, child=%v", parent1.Genes, parent2.Genes, child.Genes)
}

func TestMutate(t *testing.T) {
	original := []int{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	ind := Individual{Genes: original}
	ind.Mutate()

	for _, gene := range ind.Genes {
		if gene != 0 && gene != 1 {
			t.Errorf("expected gene to be 0 or 1, got %d", gene)
		}
	}
	t.Logf("Mutated individual: original=%v, mutated=%v", original, ind.Genes)
}

func TestTournamentSelection(t *testing.T) {
	population := []Individual{
		{Genes: []int{1, 1, 1}, Fitness: 3},
		{Genes: []int{0, 0, 0}, Fitness: 0},
		{Genes: []int{1, 1, 0}, Fitness: 2},
	}

	selected := TournamentSelection(population)
	if selected.Fitness < 0 || selected.Fitness > 3 {
		t.Errorf("expected fitness between 0 and 3, got %d", selected.Fitness)
	}
	t.Logf("Tournament selection: selected individual with fitness=%d, genes=%v", selected.Fitness, selected.Genes)
}

func TestRunGeneticAlgorithm(t *testing.T) {
	result := RunGeneticAlgorithm()

	if len(result.Genes) != targetLen {
		t.Errorf("expected genes length %d, got %d", targetLen, len(result.Genes))
	}

	if result.Fitness < 0 || result.Fitness > targetLen {
		t.Errorf("expected fitness between 0 and %d, got %d", targetLen, result.Fitness)
	}

	for _, gene := range result.Genes {
		if gene != 0 && gene != 1 {
			t.Errorf("expected gene to be 0 or 1, got %d", gene)
		}
	}

	t.Logf("Final result: fitness=%d/%d, genes=%v", result.Fitness, targetLen, result.Genes)
}
