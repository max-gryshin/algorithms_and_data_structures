package concurrency

import (
	"context"
	"time"
)

type RateLimiterTokenBucket struct {
	limiter chan struct{}
}

func NewRateLimiterTokenBucket(ctx context.Context, limit int, interval time.Duration) *RateLimiterTokenBucket {
	ticker := time.NewTicker(interval)
	rl := &RateLimiterTokenBucket{
		limiter: make(chan struct{}, limit),
	}
	for i := 0; i < limit; i++ {
		rl.limiter <- struct{}{}
	}
	go func() {
		defer func() {
			ticker.Stop()
		}()

		for {
			select {
			case <-ticker.C:
			loop:
				for i := 0; i < limit; i++ {
					select {
					case rl.limiter <- struct{}{}:
					default:
						break loop
					}
				}
			case <-ctx.Done():
				return
			}
		}
	}()

	return rl
}

func (rl *RateLimiterTokenBucket) Allow() bool {
	select {
	case <-rl.limiter:
		return true
	default:
		return false
	}
}
