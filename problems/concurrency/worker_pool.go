package concurrency

import (
	"context"
	"fmt"
	"sync"
	"time"
)

type WorkerPool struct {
	Val <-chan int
	Res chan int
	Err chan error
}

func NewTask(val <-chan int, res chan int, err chan error) *WorkerPool {
	return &WorkerPool{
		Val: val,
		Res: res,
		Err: err,
	}
}

func RunWorkers(ctx context.Context, jobs <-chan int, workers int) {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	var wg sync.WaitGroup
	// Buffering makes sense when:
	// 1. The producer and consumer are comparable in speed, but one of them has spikes—the buffer smooths them out.
	// 2. The consumer is faster than the producer—then the buffer is empty, no harm, but little benefit.
	// 3. The number of context switches needs to be reduced—rendezvous forces the scheduler to reassign control on every send/recv, and the buffer creates a batching effect.
	resCh := make(chan int, workers)
	errCh := make(chan error, 100)
	task := NewTask(jobs, resCh, errCh)

	var wgErr sync.WaitGroup
	wgErr.Add(1)
	go func() {
		defer wgErr.Done()
		for err := range task.Err {
			fmt.Println("received error", err)
		}
	}()
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(ctx context.Context, task *WorkerPool) {
			defer wg.Done()
			process := func(job int) (int, error) {
				defer func() {
					if err := recover(); err != nil {
						select {
						case task.Err <- fmt.Errorf("recovering from panic; error: %s", err):
						case <-ctx.Done():
						}
					}
				}()
				return Process(job)
			}

			for {
				select {
				case <-ctx.Done():
					return
				case v, ok := <-task.Val:
					if !ok {
						return
					}
					res, err := process(v)
					select {
					case <-ctx.Done():
						return
					case task.Res <- res:
					}
					if err != nil {
						select {
						case <-ctx.Done():
							return
						case task.Err <- err:
						}
					}
				}
			}
		}(ctx, task)
	}
	var wgRes sync.WaitGroup
	for i := 0; i < workers; i++ {
		wgRes.Add(1)
		go func(ch <-chan int) {
			defer wgRes.Done()
			for res := range ch {
				fmt.Println(res)
				time.Sleep(50 * time.Millisecond)
			}
		}(task.Res)
	}
	wg.Wait()
	close(task.Res)
	wgRes.Wait()
	close(task.Err)
	wgErr.Wait()

}

func Process(job int) (int, error) {
	time.Sleep(50 * time.Millisecond)
	return job * 2, nil
}
