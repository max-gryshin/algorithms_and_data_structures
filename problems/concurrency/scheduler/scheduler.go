package scheduler

import (
	"container/heap"
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"golang.org/x/sync/errgroup"
)

type TaskState int

const (
	StatePending TaskState = iota
	StateRunning
	StateCompleted
	StateFailed
)

type Color int

const (
	White Color = iota // Unvisited
	Gray               // Visiting (in recursion stack)
	Black              // Visited
)

type TaskFn[T any] func(ctx context.Context) (T, error)

type Task struct {
	ID              string
	Fn              TaskFn[any]
	Deps            []string // Список ID задач, от которых зависит эта задача
	Priority        int      // Статический приоритет (1-100, по умолчанию 50)
	DynamicPriority int      // Динамический приоритет (зависит от DAG)
	State           TaskState
	depsCount       int     // Сколько счетчиков невыполненных зависимостей осталось
	children        []*Task // Список задач, которые ждут выполнения текущей
	downstreamCount int     // Сколько задач транзитивно зависит от этой
}

type Scheduler[T any] struct {
	tasks              map[string]*Task
	mu                 sync.Mutex
	concurrency        int
	taskCfg            TaskConfig
	downstreamCounts   map[string]int // Кэш для downstreamCount каждой задачи
	downstreamCountsMu sync.Mutex
}

type Result[T any] struct {
	Index int
	Value T
	Err   error
}

type TaskConfig struct {
	MaxRetries     int           // K попыток повтора (всего MaxRetries + 1 запуск)
	InitialBackoff time.Duration // Начальный backoff (например, 100ms)
	Timeout        time.Duration // Индивидуальный таймаут на 1 попытку
}

// NewScheduler создает новый экземпляра планировщика.
func NewScheduler[T any](concurrency int, cfg TaskConfig) *Scheduler[T] {
	if concurrency <= 0 {
		concurrency = 1
	}
	return &Scheduler[T]{
		tasks:            make(map[string]*Task),
		concurrency:      concurrency,
		taskCfg:          cfg,
		downstreamCounts: make(map[string]int),
	}
}

// Add добавляет задачу в очереди и проверяет на циклические зависимости.
// priority опциональный параметр (по умолчанию 50). Если передано <= 0, используется 50.
func (s *Scheduler[T]) Add(id string, deps []string, fn TaskFn[any], priority ...int) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	p := 50
	if len(priority) > 0 && priority[0] > 0 {
		p = priority[0]
	}

	s.tasks[id] = &Task{
		ID:        id,
		Fn:        fn,
		Deps:      deps,
		Priority:  p,
		State:     StatePending,
		depsCount: len(deps),
	}
	return s.validateDAG()
}

// validateDAG проверяет граф на циклы алгоритмом DFS.
func (s *Scheduler[T]) validateDAG() error {
	visited := make(map[string]Color)

	var dfs func(id string) error
	dfs = func(id string) error {
		visited[id] = Gray

		task, exists := s.tasks[id]
		if !exists {
			return fmt.Errorf("missing task %s", id)
		}

		for _, depID := range task.Deps {
			state := visited[depID]
			if state == Gray {
				return fmt.Errorf("cycle detected: task %s -> %s", id, depID)
			}
			if state == White {
				if err := dfs(depID); err != nil {
					return err
				}
			}
		}

		visited[id] = Black
		return nil
	}

	for id := range s.tasks {
		if visited[id] == White {
			if err := dfs(id); err != nil {
				return err
			}
		}
	}

	return nil
}

// calculateDownstreamCount вычисляет количество задач, которые транзитивно зависят от задачи с заданным ID.
// Используется BFS для прохода по графу children.
func (s *Scheduler[T]) calculateDownstreamCount(taskID string) int {
	task, exists := s.tasks[taskID]
	if !exists {
		return 0
	}
	visited := make(map[string]bool)
	queue := []*Task{task}
	count := 0

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		for _, child := range current.children {
			if !visited[child.ID] {
				visited[child.ID] = true
				count++
				queue = append(queue, child)
			}
		}
	}

	return count
}

// calculateDynamicPriority вычисляет динамический приоритет на основе количества дочерних задач.
// динамический приоритет = downstreamCount * 5
func (s *Scheduler[T]) calculateDynamicPriority(taskID string) int {
	s.downstreamCountsMu.Lock()
	count, exists := s.downstreamCounts[taskID]
	s.downstreamCountsMu.Unlock()

	if !exists {
		s.mu.Lock()
		count = s.calculateDownstreamCount(taskID)
		s.mu.Unlock()

		s.downstreamCountsMu.Lock()
		s.downstreamCounts[taskID] = count
		s.downstreamCountsMu.Unlock()
	}

	priority := count * 5

	s.mu.Lock()
	if task, exists := s.tasks[taskID]; exists {
		task.DynamicPriority = priority
	}
	s.mu.Unlock()

	return priority
}

// getFinalPriority возвращает финальный приоритет задачи.
// формула: FinalPriority = StaticPriority*2 + DynamicPriority
func (s *Scheduler[T]) getFinalPriority(task *Task) int {
	return task.Priority*2 + task.DynamicPriority
}

func (s *Scheduler[T]) Run(ctx context.Context, concurrency int) error {
	s.mu.Lock()

	// 1. Построение обратных связей (кто зависит от меня)
	for _, task := range s.tasks {
		for _, depID := range task.Deps {
			parent, exists := s.tasks[depID]
			if !exists {
				s.mu.Unlock()
				return fmt.Errorf("task %s depends on missing task %s", task.ID, depID)
			}
			parent.children = append(parent.children, task)
		}
	}

	s.mu.Unlock()

	// 2. Вычисляем динамические приоритеты для всех задач
	for taskID := range s.tasks {
		s.calculateDynamicPriority(taskID)
	}

	g, gCtx := errgroup.WithContext(ctx)
	if concurrency > 0 {
		g.SetLimit(concurrency)
	}

	// 4. Главный цикл оркестрации
	var wg sync.WaitGroup
	wg.Add(len(s.tasks))

	pq := make(PriorityQueue, 0, len(s.tasks))
	pqMu := sync.Mutex{}

	// Отправляем начальные готовые задачи в очередь приоритетов
	s.mu.Lock()
	for _, task := range s.tasks {
		if task.depsCount == 0 {
			heap.Push(&pq, &PrioritizedTask{
				Priority: s.getFinalPriority(task),
				TaskID:   task.ID,
			})
		}
	}
	s.mu.Unlock()

	dispatcherDone := make(chan struct{})

	go func() {
		defer close(dispatcherDone)

		for {
			pqMu.Lock()
			if len(pq) == 0 {
				pqMu.Unlock()
				select {
				case <-gCtx.Done():
					return
				default:
					time.Sleep(1 * time.Millisecond)
					continue
				}
			}

			prioritizedTask := heap.Pop(&pq).(*PrioritizedTask)
			pqMu.Unlock()

			s.mu.Lock()
			task, exists := s.tasks[prioritizedTask.TaskID]
			s.mu.Unlock()

			if !exists {
				continue
			}

			g.Go(func() error {
				defer wg.Done()

				if err := s.executeWithTaskRetry(gCtx, task.Fn); err != nil {
					task.State = StateFailed
					return fmt.Errorf("task %s failed: %w", task.ID, err)
				}

				task.State = StateCompleted

				var readyChildren []*Task
				s.mu.Lock()
				for _, child := range task.children {
					child.depsCount--
					if child.depsCount == 0 {
						child.State = StateRunning
						readyChildren = append(readyChildren, child)
					}
				}
				s.mu.Unlock()

				// Динамически вычисляем приоритеты и добавляем в очередь
				for _, child := range readyChildren {
					s.calculateDynamicPriority(child.ID)

					pqMu.Lock()
					heap.Push(&pq, &PrioritizedTask{
						Priority: s.getFinalPriority(child),
						TaskID:   child.ID,
					})
					pqMu.Unlock()
				}

				return nil
			})
		}
	}()

	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		return g.Wait()
	case <-gCtx.Done():
		return gCtx.Err()
	}
}

// Логика Retry + Exponential Backoff + Timeout
// executeWithTaskRetry выполняет TaskFn[any] с ретрайами, используется в Run для работы с зависимостями.
func (s *Scheduler[T]) executeWithTaskRetry(ctx context.Context, fn TaskFn[any]) error {
	backoff := s.taskCfg.InitialBackoff

	for attempt := 0; attempt <= s.taskCfg.MaxRetries; attempt++ {
		if err := ctx.Err(); err != nil {
			return err
		}

		_, err := s.executeWithTaskTimeout(ctx, fn)
		if err == nil {
			return nil
		}

		if attempt == s.taskCfg.MaxRetries {
			return fmt.Errorf("all retries (%d) failed, last error: %w", s.taskCfg.MaxRetries, err)
		}

		select {
		case <-time.After(backoff):
			backoff *= 2
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	return errors.New("unexpected exit from retry loop")
}

// executeWithTaskTimeout оборачивает вызов TaskFn[any] в контекст с таймаутом.
func (s *Scheduler[T]) executeWithTaskTimeout(ctx context.Context, fn TaskFn[any]) (any, error) {
	taskCtx := ctx
	var cancel context.CancelFunc

	if s.taskCfg.Timeout > 0 {
		taskCtx, cancel = context.WithTimeout(ctx, s.taskCfg.Timeout)
		defer cancel()
	}

	type runRes struct {
		val any
		err error
	}

	ch := make(chan runRes, 1)

	go func() {
		val, err := fn(taskCtx)
		ch <- runRes{val: val, err: err}
	}()

	select {
	case res := <-ch:
		return res.val, res.err
	case <-taskCtx.Done():
		return nil, taskCtx.Err()
	}
}
