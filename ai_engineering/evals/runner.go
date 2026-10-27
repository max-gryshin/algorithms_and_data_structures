package evals

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// Target is the system under test: prompt chain, RAG pipeline, agent — anything
// that maps an input to an output.
type Target func(ctx context.Context, input string) (string, error)

// Result is the outcome of one case.
type Result struct {
	Case    Case          `json:"case"`
	Output  string        `json:"output"`
	Grade   Grade         `json:"grade"`
	Err     string        `json:"error,omitempty"` // target or grader failure; counts as a fail
	Latency time.Duration `json:"latency_ns"`
}

// Run holds the results of one dataset run, in dataset order.
type Run struct {
	Name    string   `json:"name"`
	Results []Result `json:"results"`
}

// Runner evaluates a dataset with a bounded pool of workers.
type Runner struct {
	Target  Target
	Grader  Grader
	Workers int           // concurrent cases (respect the API rate limit), default 4
	Timeout time.Duration // per case; 0 = no timeout
}

// Run evaluates every case. A failing case never stops the run; cancelling ctx does.
func (r *Runner) Run(ctx context.Context, name string, ds *Dataset) (*Run, error) {
	workers := r.Workers
	if workers <= 0 {
		workers = 4
	}
	results := make([]Result, len(ds.Cases))
	jobs := make(chan int)
	var wg sync.WaitGroup
	for range workers {
		wg.Go(func() {
			for i := range jobs {
				results[i] = r.one(ctx, ds.Cases[i])
			}
		})
	}
	for i := range ds.Cases {
		select {
		case jobs <- i:
		case <-ctx.Done():
			close(jobs)
			wg.Wait()
			return nil, ctx.Err()
		}
	}
	close(jobs)
	wg.Wait()
	return &Run{Name: name, Results: results}, ctx.Err()
}

func (r *Runner) one(ctx context.Context, c Case) Result {
	if r.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, r.Timeout)
		defer cancel()
	}
	res := Result{Case: c}
	start := time.Now()
	out, err := r.Target(ctx, c.Input)
	res.Latency = time.Since(start)
	res.Output = out
	if err != nil {
		res.Err = fmt.Sprintf("target: %v", err)
		return res
	}
	g, err := r.Grader.Grade(ctx, c, out)
	if err != nil {
		res.Err = fmt.Sprintf("grader: %v", err)
		return res
	}
	res.Grade = g
	return res
}
