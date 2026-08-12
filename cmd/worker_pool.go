package main

import (
	"algorithms_and_data_structures/problems/concurrency"
	"context"
	"os"
	"os/signal"
	"syscall"
)

func main() {
	ctx, cancelFunc := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancelFunc()
	concurrency.RunWorkers(ctx, jobsChan([]int{1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 9, 23, 234, 54, 54345}), 5)

}

func jobsChan(jobs []int) <-chan int {
	ch := make(chan int)
	go func(ch chan<- int) {
		defer close(ch)
		for _, job := range jobs {
			ch <- job
		}
	}(ch)

	return ch
}
