package optimization

import (
	"testing"
)

func TestEvaluateMLModel(t *testing.T) {
	hp := Hyperparameters{
		MaxDepth:     6,
		LearningRate: 0.15,
		Threshold:    0.75,
	}
	fitness := evaluateMLModel(&hp)

	if fitness <= 0 || fitness > 100 {
		t.Errorf("expected fitness between 0 and 100, got %.2f", fitness)
	}
	t.Logf("Evaluated model: MaxDepth=%d, LR=%.4f, Threshold=%.4f, Fitness=%.2f%%",
		hp.MaxDepth, hp.LearningRate, hp.Threshold, fitness)
}

func TestRandomHP(t *testing.T) {
	hp := randomHP()

	if hp.MaxDepth < 2 || hp.MaxDepth > 10 {
		t.Errorf("expected MaxDepth between 2 and 10, got %d", hp.MaxDepth)
	}
	if hp.LearningRate < 0.01 || hp.LearningRate > 0.3 {
		t.Errorf("expected LearningRate between 0.01 and 0.3, got %.4f", hp.LearningRate)
	}
	if hp.Threshold < 0.4 || hp.Threshold > 0.9 {
		t.Errorf("expected Threshold between 0.4 and 0.9, got %.4f", hp.Threshold)
	}

	t.Logf("Generated random hyperparameters: MaxDepth=%d, LR=%.4f, Threshold=%.4f",
		hp.MaxDepth, hp.LearningRate, hp.Threshold)
}

func TestHPCrossover(t *testing.T) {
	p1 := Hyperparameters{
		MaxDepth:     8,
		LearningRate: 0.2,
		Threshold:    0.8,
	}
	p2 := Hyperparameters{
		MaxDepth:     4,
		LearningRate: 0.1,
		Threshold:    0.6,
	}

	child := HPCrossover(p1, p2)

	if (child.MaxDepth != p1.MaxDepth && child.MaxDepth != p2.MaxDepth) ||
		child.MaxDepth < 2 || child.MaxDepth > 10 {
		t.Errorf("invalid child MaxDepth from crossover: %d", child.MaxDepth)
	}
	if child.LearningRate < 0.01 || child.LearningRate > 0.3 {
		t.Errorf("invalid child LearningRate from crossover: %.4f", child.LearningRate)
	}
	if child.Threshold < 0.4 || child.Threshold > 0.9 {
		t.Errorf("invalid child Threshold from crossover: %.4f", child.Threshold)
	}

	t.Logf("Crossover: p1=(D%d,LR%.3f,T%.2f), p2=(D%d,LR%.3f,T%.2f), child=(D%d,LR%.3f,T%.2f)",
		p1.MaxDepth, p1.LearningRate, p1.Threshold,
		p2.MaxDepth, p2.LearningRate, p2.Threshold,
		child.MaxDepth, child.LearningRate, child.Threshold)
}

func TestHPMutate(t *testing.T) {
	hp := Hyperparameters{
		MaxDepth:     6,
		LearningRate: 0.15,
		Threshold:    0.75,
	}
	original := Hyperparameters{
		MaxDepth:     hp.MaxDepth,
		LearningRate: hp.LearningRate,
		Threshold:    hp.Threshold,
	}

	hp.HPMutate()

	if hp.MaxDepth < 2 || hp.MaxDepth > 10 {
		t.Errorf("MaxDepth out of bounds after mutation: %d", hp.MaxDepth)
	}
	if hp.LearningRate < 0.01 || hp.LearningRate > 0.3 {
		t.Errorf("LearningRate out of bounds after mutation: %.4f", hp.LearningRate)
	}
	if hp.Threshold < 0.4 || hp.Threshold > 0.9 {
		t.Errorf("Threshold out of bounds after mutation: %.4f", hp.Threshold)
	}

	t.Logf("Mutate: original=(D%d,LR%.3f,T%.2f), mutated=(D%d,LR%.3f,T%.2f)",
		original.MaxDepth, original.LearningRate, original.Threshold,
		hp.MaxDepth, hp.LearningRate, hp.Threshold)
}

func TestRunAutoMLAlgorithm(t *testing.T) {
	result := RunAutoMLAlgorithm()

	if result.MaxDepth < 2 || result.MaxDepth > 10 {
		t.Errorf("result MaxDepth out of bounds: %d", result.MaxDepth)
	}
	if result.LearningRate < 0.01 || result.LearningRate > 0.3 {
		t.Errorf("result LearningRate out of bounds: %.4f", result.LearningRate)
	}
	if result.Threshold < 0.4 || result.Threshold > 0.9 {
		t.Errorf("result Threshold out of bounds: %.4f", result.Threshold)
	}
	if result.Fitness <= 0 || result.Fitness > 100 {
		t.Errorf("result Fitness out of bounds: %.2f", result.Fitness)
	}

	t.Logf("AutoML Result: MaxDepth=%d, LearningRate=%.4f, Threshold=%.4f, Fitness=%.2f%%",
		result.MaxDepth, result.LearningRate, result.Threshold, result.Fitness)
}
