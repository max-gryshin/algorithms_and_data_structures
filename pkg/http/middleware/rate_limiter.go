package middleware

import (
	rl "algorithms_and_data_structures/problems/concurrency/rate_limiter"
	"context"
	"net/http"
)

type RateLimiter struct {
	rateLimiter rl.RateLimiter
}

func NewRateLimiter(ctx context.Context) *RateLimiter {
	return &RateLimiter{
		rateLimiter: rl.NewLeakyBucket(ctx, rl.DefaultLimit, rl.DefaultInterval),
	}
}

func (rl *RateLimiter) RateLimiterHandler(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if rl.rateLimiter.Allow() {
			next.ServeHTTP(w, r)
			return
		}
		w.WriteHeader(http.StatusTooManyRequests)
	})
}
