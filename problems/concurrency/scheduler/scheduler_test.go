package scheduler

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestBasicExecution(t *testing.T) {
	cfg := TaskConfig{
		MaxRetries:     0,
		InitialBackoff: 0,
		Timeout:        0,
	}
	scheduler := NewScheduler[int](2, cfg)

	order := []string{}
	var mu sync.Mutex

	// Task A (no deps)
	scheduler.Add("A", []string{}, func(ctx context.Context) (any, error) {
		mu.Lock()
		order = append(order, "A")
		mu.Unlock()
		return 1, nil
	})

	// Task B (depends on A)
	scheduler.Add("B", []string{"A"}, func(ctx context.Context) (any, error) {
		mu.Lock()
		order = append(order, "B")
		mu.Unlock()
		return 2, nil
	})

	err := scheduler.Run(context.Background(), 2)
	if err != nil {
		t.Fatalf("Run() failed: %v", err)
	}

	if len(order) != 2 || order[0] != "A" || order[1] != "B" {
		t.Errorf("Expected execution order [A, B], got %v", order)
	}
}

func TestDAGWithMultipleDependencies(t *testing.T) {
	cfg := TaskConfig{MaxRetries: 0, InitialBackoff: 0, Timeout: 0}
	scheduler := NewScheduler[int](4, cfg)

	order := []string{}
	var mu sync.Mutex

	// A (no deps)
	scheduler.Add("A", []string{}, func(ctx context.Context) (any, error) {
		mu.Lock()
		order = append(order, "A")
		mu.Unlock()
		return 1, nil
	})

	// B, C (depend on A)
	scheduler.Add("B", []string{"A"}, func(ctx context.Context) (any, error) {
		mu.Lock()
		order = append(order, "B")
		mu.Unlock()
		return 2, nil
	})

	scheduler.Add("C", []string{"A"}, func(ctx context.Context) (any, error) {
		mu.Lock()
		order = append(order, "C")
		mu.Unlock()
		return 3, nil
	})

	// D (depends on B and C)
	scheduler.Add("D", []string{"B", "C"}, func(ctx context.Context) (any, error) {
		mu.Lock()
		order = append(order, "D")
		mu.Unlock()
		return 4, nil
	})

	err := scheduler.Run(context.Background(), 4)
	if err != nil {
		t.Fatalf("Run() failed: %v", err)
	}

	if len(order) != 4 {
		t.Fatalf("Expected 4 tasks executed, got %d", len(order))
	}

	// A должна быть первой
	if order[0] != "A" {
		t.Errorf("Task A should be first, got %v", order[0])
	}

	// B и C должны быть после A
	bIdx, cIdx := -1, -1
	for i, task := range order {
		if task == "B" {
			bIdx = i
		}
		if task == "C" {
			cIdx = i
		}
	}

	if bIdx < 1 || cIdx < 1 {
		t.Errorf("B and C should come after A")
	}

	// D должна быть последней
	if order[3] != "D" {
		t.Errorf("Task D should be last, got %v", order[3])
	}
}

func TestCycleDetection(t *testing.T) {
	cfg := TaskConfig{MaxRetries: 0, InitialBackoff: 0, Timeout: 0}
	scheduler := NewScheduler[int](2, cfg)

	// Add A
	scheduler.Add("A", []string{}, func(ctx context.Context) (any, error) { return 1, nil })

	// Add B depends on A
	scheduler.Add("B", []string{"A"}, func(ctx context.Context) (any, error) { return 2, nil })

	// Try to add A with dependency on B - should detect cycle
	err := scheduler.Add("A", []string{"B"}, func(ctx context.Context) (any, error) { return 1, nil })
	if err == nil {
		t.Fatal("Expected cycle detection error, got nil")
	}

	if !contains(err.Error(), "cycle") {
		t.Errorf("Expected cycle error, got: %v", err)
	}
}

func TestStaticPriorityExecution(t *testing.T) {
	cfg := TaskConfig{MaxRetries: 0, InitialBackoff: 0, Timeout: 0}
	scheduler := NewScheduler[int](1, cfg) // Single worker

	order := []string{}
	var mu sync.Mutex

	// Task Low (Priority 10) - added first but should execute last
	scheduler.Add("Low", []string{}, func(ctx context.Context) (any, error) {
		mu.Lock()
		order = append(order, "Low")
		mu.Unlock()
		time.Sleep(10 * time.Millisecond)
		return 1, nil
	}, 10)

	// Task High (Priority 100) - added second but should execute first
	scheduler.Add("High", []string{}, func(ctx context.Context) (any, error) {
		mu.Lock()
		order = append(order, "High")
		mu.Unlock()
		time.Sleep(10 * time.Millisecond)
		return 2, nil
	}, 100)

	err := scheduler.Run(context.Background(), 1)
	if err != nil {
		t.Fatalf("Run() failed: %v", err)
	}

	// High should be first due to higher priority
	if len(order) != 2 || order[0] != "High" || order[1] != "Low" {
		t.Errorf("Expected [High, Low], got %v", order)
	}
}

func TestDynamicPriorityExecution(t *testing.T) {
	cfg := TaskConfig{MaxRetries: 0, InitialBackoff: 0, Timeout: 0}
	scheduler := NewScheduler[int](1, cfg) // Single worker

	order := []string{}
	var mu sync.Mutex

	// Task A (no deps, default priority 50) - but 3 tasks depend on it
	scheduler.Add("A", []string{}, func(ctx context.Context) (any, error) {
		mu.Lock()
		order = append(order, "A")
		mu.Unlock()
		return 1, nil
	})

	// Task B (default priority 50) - only 1 task depends on it
	scheduler.Add("B", []string{}, func(ctx context.Context) (any, error) {
		mu.Lock()
		order = append(order, "B")
		mu.Unlock()
		return 2, nil
	})

	// Tasks depending on A
	scheduler.Add("A1", []string{"A"}, func(ctx context.Context) (any, error) {
		mu.Lock()
		order = append(order, "A1")
		mu.Unlock()
		return 3, nil
	})

	scheduler.Add("A2", []string{"A"}, func(ctx context.Context) (any, error) {
		mu.Lock()
		order = append(order, "A2")
		mu.Unlock()
		return 4, nil
	})

	scheduler.Add("A3", []string{"A"}, func(ctx context.Context) (any, error) {
		mu.Lock()
		order = append(order, "A3")
		mu.Unlock()
		return 5, nil
	})

	// Task depending on B
	scheduler.Add("B1", []string{"B"}, func(ctx context.Context) (any, error) {
		mu.Lock()
		order = append(order, "B1")
		mu.Unlock()
		return 6, nil
	})

	err := scheduler.Run(context.Background(), 1)
	if err != nil {
		t.Fatalf("Run() failed: %v", err)
	}

	// A should execute before B because it has higher dynamic priority (3 downstream vs 1)
	aIdx, bIdx := -1, -1
	for i, task := range order {
		if task == "A" {
			aIdx = i
		}
		if task == "B" {
			bIdx = i
		}
	}

	if aIdx == -1 || bIdx == -1 {
		t.Fatalf("Tasks A and B not found in execution order")
	}

	if aIdx > bIdx {
		t.Errorf("Task A (higher dynamic priority) should execute before B, got order: %v", order)
	}
}

func TestTaskFailure(t *testing.T) {
	cfg := TaskConfig{MaxRetries: 0, InitialBackoff: 0, Timeout: 0}
	scheduler := NewScheduler[int](2, cfg)

	scheduler.Add("A", []string{}, func(ctx context.Context) (any, error) {
		return 1, nil
	})

	scheduler.Add("B", []string{}, func(ctx context.Context) (any, error) {
		return nil, errors.New("task B failed")
	})

	err := scheduler.Run(context.Background(), 2)
	if err == nil {
		t.Fatal("Expected error from failed task, got nil")
	}

	// Ошибка может быть "task B failed" или "context canceled"
	// (в зависимости от того когда errgroup отменит контекст)
	if !contains(err.Error(), "failed") && !contains(err.Error(), "canceled") {
		t.Errorf("Expected error about task failure, got: %v", err)
	}
}

func TestContextCancellation(t *testing.T) {
	cfg := TaskConfig{MaxRetries: 0, InitialBackoff: 0, Timeout: 0}
	scheduler := NewScheduler[int](2, cfg)

	executed := atomic.Int32{}

	scheduler.Add("A", []string{}, func(ctx context.Context) (any, error) {
		executed.Add(1)
		return 1, nil
	})

	scheduler.Add("B", []string{"A"}, func(ctx context.Context) (any, error) {
		time.Sleep(100 * time.Millisecond)
		executed.Add(1)
		return 2, nil
	})

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	_ = scheduler.Run(ctx, 2)

	// Should have executed at least A
	if executed.Load() < 1 {
		t.Errorf("Expected at least 1 task executed, got %d", executed.Load())
	}
}

func TestRetryLogic(t *testing.T) {
	cfg := TaskConfig{
		MaxRetries:     2,
		InitialBackoff: 10 * time.Millisecond,
		Timeout:        1 * time.Second,
	}
	scheduler := NewScheduler[int](1, cfg)

	attempts := atomic.Int32{}

	scheduler.Add("A", []string{}, func(ctx context.Context) (any, error) {
		attempts.Add(1)
		if attempts.Load() < 3 {
			return nil, errors.New("retry me")
		}
		return 42, nil
	})

	err := scheduler.Run(context.Background(), 1)
	if err != nil {
		t.Fatalf("Run() failed: %v", err)
	}

	if attempts.Load() != 3 {
		t.Errorf("Expected 3 attempts (1 initial + 2 retries), got %d", attempts.Load())
	}
}

func TestTimeoutExecution(t *testing.T) {
	cfg := TaskConfig{
		MaxRetries:     0,
		InitialBackoff: 0,
		Timeout:        50 * time.Millisecond,
	}
	scheduler := NewScheduler[int](1, cfg)

	scheduler.Add("A", []string{}, func(ctx context.Context) (any, error) {
		return 1, nil
	})

	// Task that exceeds timeout
	scheduler.Add("B", []string{"A"}, func(ctx context.Context) (any, error) {
		select {
		case <-time.After(200 * time.Millisecond):
			return 2, nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	})

	err := scheduler.Run(context.Background(), 1)
	if err == nil {
		t.Fatal("Expected timeout error, got nil")
	}
}

func TestConcurrency(t *testing.T) {
	cfg := TaskConfig{MaxRetries: 0, InitialBackoff: 0, Timeout: 0}
	scheduler := NewScheduler[int](4, cfg)

	concurrent := atomic.Int32{}
	maxConcurrent := atomic.Int32{}
	var mu sync.Mutex

	for i := 0; i < 10; i++ {
		id := string(rune('A' + i))
		scheduler.Add(id, []string{}, func(ctx context.Context) (any, error) {
			concurrent.Add(1)
			curr := concurrent.Load()

			mu.Lock()
			if curr > maxConcurrent.Load() {
				maxConcurrent.Store(curr)
			}
			mu.Unlock()

			time.Sleep(10 * time.Millisecond)
			concurrent.Add(-1)
			return 1, nil
		})
	}

	err := scheduler.Run(context.Background(), 4)
	if err != nil {
		t.Fatalf("Run() failed: %v", err)
	}

	// Max concurrent should be <= 4
	if maxConcurrent.Load() > 4 {
		t.Errorf("Expected max concurrency <= 4, got %d", maxConcurrent.Load())
	}
}

func TestComplexDAG(t *testing.T) {
	cfg := TaskConfig{MaxRetries: 0, InitialBackoff: 0, Timeout: 0}
	scheduler := NewScheduler[int](4, cfg)

	/*
		  A
		 / \
		B   C
		 \ / \
		  D   E
		  |
		  F
	*/

	order := []string{}
	var mu sync.Mutex

	tasks := map[string][]string{
		"A": {},
		"B": {"A"},
		"C": {"A"},
		"D": {"B", "C"},
		"E": {"C"},
		"F": {"D"},
	}

	for id, deps := range tasks {
		taskID := id
		scheduler.Add(taskID, deps, func(ctx context.Context) (any, error) {
			mu.Lock()
			order = append(order, taskID)
			mu.Unlock()
			return 1, nil
		})
	}

	err := scheduler.Run(context.Background(), 4)
	if err != nil {
		t.Fatalf("Run() failed: %v", err)
	}

	if len(order) != 6 {
		t.Fatalf("Expected 6 tasks, got %d", len(order))
	}

	// Verify dependencies are respected
	validateDependencies(t, order, tasks)
}

func validateDependencies(t *testing.T, order []string, deps map[string][]string) {
	indices := make(map[string]int)
	for i, task := range order {
		indices[task] = i
	}

	for task, taskDeps := range deps {
		taskIdx := indices[task]
		for _, dep := range taskDeps {
			depIdx := indices[dep]
			if depIdx > taskIdx {
				t.Errorf("Task %s depends on %s, but %s executed after %s", task, dep, dep, task)
			}
		}
	}
}

func TestMissingDependency(t *testing.T) {
	cfg := TaskConfig{MaxRetries: 0, InitialBackoff: 0, Timeout: 0}
	scheduler := NewScheduler[int](2, cfg)

	scheduler.Add("A", []string{}, func(ctx context.Context) (any, error) {
		return 1, nil
	})

	// Try to add task with non-existent dependency
	scheduler.Add("B", []string{"NonExistent"}, func(ctx context.Context) (any, error) {
		return 2, nil
	})

	err := scheduler.Run(context.Background(), 2)
	if err == nil {
		t.Fatal("Expected error for missing dependency, got nil")
	}

	if !contains(err.Error(), "missing") {
		t.Errorf("Expected 'missing' in error, got: %v", err)
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && s[:len(substr)] == substr || len(s) > len(substr) && contains(s[1:], substr)
}
