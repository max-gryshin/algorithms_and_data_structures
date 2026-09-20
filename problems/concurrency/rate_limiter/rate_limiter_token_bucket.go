package rate_limiter

import (
	"context"
	"time"
)

type TokenBucket struct {
	limiter chan struct{}
}

func NewTokenBucket(ctx context.Context, limit int, interval time.Duration) *TokenBucket {
	ticker := time.NewTicker(interval)
	rl := &TokenBucket{
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

func (rl *TokenBucket) Allow() bool {
	select {
	case <-rl.limiter:
		return true
	default:
		return false
	}
}
