package rate_limiter

import (
	"context"
	"time"
)

const (
	DefaultLimit    = 1000
	DefaultInterval = 1 * time.Second
)

// LeakyBucket leaky-bucket-like limiter
type LeakyBucket struct {
	limiter chan struct{}
}

func NewLeakyBucket(ctx context.Context, limit int, interval time.Duration) *LeakyBucket {
	ticker := time.NewTicker(interval / time.Duration(limit))
	rl := &LeakyBucket{
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

func (r *LeakyBucket) Allow() bool {
	select {
	case r.limiter <- struct{}{}:
		return true
	default:
		return false
	}
}
