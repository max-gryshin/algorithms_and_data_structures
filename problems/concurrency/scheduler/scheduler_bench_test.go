package scheduler

import (
	"context"
	"testing"
)

func BenchmarkScheduler(b *testing.B) {
	cfg := TaskConfig{MaxRetries: 0, InitialBackoff: 0, Timeout: 0}

	for i := 0; i < b.N; i++ {
		scheduler := NewScheduler[int](4, cfg)

		for j := 0; j < 100; j++ {
			id := string(rune('A' + (j % 26)))
			scheduler.Add(id+string(rune('0'+(j/26))), []string{}, func(ctx context.Context) (any, error) {
				return 1, nil
			})
		}

		_ = scheduler.Run(context.Background(), 4)
	}
}
