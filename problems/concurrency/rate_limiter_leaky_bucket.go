package concurrency

import (
	"context"
	"time"
)

// RateLimiterLeakyBucket leaky-bucket-like limiter
type RateLimiterLeakyBucket struct {
	limiter chan struct{}
}

func NewRateLimiterLeakyBucket(ctx context.Context, limit int, interval time.Duration) *RateLimiterLeakyBucket {
	ticker := time.NewTicker(interval / time.Duration(limit))
	rl := &RateLimiterLeakyBucket{
		limiter: make(chan struct{}, limit),
	}
	go func() {
		defer func() {
			ticker.Stop()
		}()
	loop:
		for {
			select {
			case <-ticker.C:
				select {
				case <-rl.limiter:
				default:
					continue
				}
			case <-ctx.Done():
				break loop
			}
		}
	}()

	return rl
}

func (r *RateLimiterLeakyBucket) Allow() bool {
	select {
	case r.limiter <- struct{}{}:
		return true
	default:
		return false
	}
}
