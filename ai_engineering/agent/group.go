package agent

import (
	"context"
	"sync"
)

// Group runs goroutines with a concurrency limit; the first error cancels the shared
// context and is returned by Wait. It is the errgroup pattern
// (golang.org/x/sync/errgroup) in 40 lines of stdlib:
//
//	g, ctx := WithContext(ctx)
//	g.SetLimit(4)
//	for _, job := range jobs {
//		g.Go(func() error { return work(ctx, job) })
//	}
//	err := g.Wait()
type Group struct {
	wg     sync.WaitGroup
	sem    chan struct{}
	once   sync.Once
	err    error
	cancel context.CancelCauseFunc
}

// WithContext returns a group and a context canceled on the first error or after Wait.
func WithContext(ctx context.Context) (*Group, context.Context) {
	ctx, cancel := context.WithCancelCause(ctx)
	return &Group{cancel: cancel}, ctx
}

// SetLimit caps the number of goroutines running at once. Call it before Go.
func (g *Group) SetLimit(n int) {
	if n > 0 {
		g.sem = make(chan struct{}, n)
	}
}

// Go runs f in a new goroutine, blocking while the limit is reached.
func (g *Group) Go(f func() error) {
	if g.sem != nil {
		g.sem <- struct{}{} // acquire a slot (backpressure on the caller)
	}
	g.wg.Go(func() {
		defer func() {
			if g.sem != nil {
				<-g.sem
			}
		}()
		if err := f(); err != nil {
			g.once.Do(func() {
				g.err = err
				if g.cancel != nil {
					g.cancel(err)
				}
			})
		}
	})
}

// Wait blocks until all goroutines finish and returns the first error.
func (g *Group) Wait() error {
	g.wg.Wait()
	if g.cancel != nil {
		g.cancel(g.err)
	}
	return g.err
}
