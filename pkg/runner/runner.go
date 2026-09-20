package runner

import (
	"context"
	"fmt"
	"runtime/debug"

	"golang.org/x/sync/errgroup"
)

// Func - Adapter func
// Example:
// Run(ctx,
//
//	Server{},
//	Func(func(ctx context.Context) error {
//	    fmt.Println("background job")
//	    return nil
//	}),
//
// )
type Func func(ctx context.Context) error

func (f Func) Run(ctx context.Context) error {
	return f(ctx)
}

type Runner interface {
	Run(context.Context) error
}

func Run(ctx context.Context, runners ...Runner) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	errGroup, errGroupCtx := errgroup.WithContext(ctx)
	for _, runner := range runners {
		errGroup.Go(func() error {
			defer func() {
				if rec := recover(); rec != nil {
					fmt.Println(fmt.Errorf("could not recover from panic %v; stack trace: %s", rec, debug.Stack()))
				}
				cancel()
			}()
			return runner.Run(errGroupCtx)
		})
	}
	errGroup.Wait()

	return nil
}
