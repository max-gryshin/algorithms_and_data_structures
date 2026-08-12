package concurrency

import (
	"sync"
	"time"
)

type RateLimiterFixedWindow struct {
	limit       int
	interval    time.Duration
	count       int
	windowStart time.Time
	mu          sync.Mutex
}

func NewFixedWindowLimiter(limit int, interval time.Duration) *RateLimiterFixedWindow {
	return &RateLimiterFixedWindow{
		limit:       limit,
		interval:    interval,
		windowStart: time.Now(),
	}
}

func (r *RateLimiterFixedWindow) Allow() bool {
	r.mu.Lock()
	defer r.mu.Unlock()

	now := time.Now()

	if now.Sub(r.windowStart) >= r.interval {
		r.windowStart = now
		r.count = 0
	}

	if r.count >= r.limit {
		return false
	}

	r.count++
	return true
}
