package main

import (
	"algorithms_and_data_structures/pkg/http"
	"algorithms_and_data_structures/pkg/runner"
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"
)

func main() {
	if err := run(); err != nil {
		fmt.Println(err)
	}
}

func run() error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	srv := http.NewServer(ctx, 8080)
	if err := runner.Run(ctx, srv); err != nil {
		return fmt.Errorf("runner returned with an error: %v", err)
	}

	return nil
}
