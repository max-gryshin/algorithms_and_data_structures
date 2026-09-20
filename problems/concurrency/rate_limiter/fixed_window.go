package rate_limiter

import (
	"sync"
	"time"
)

type FixedWindow struct {
	interval    time.Duration
	windowStart time.Time
	mu          sync.Mutex
	limit       int
	count       int
}

func NewFixedWindowLimiter(limit int, interval time.Duration) *FixedWindow {
	return &FixedWindow{
		limit:       limit,
		interval:    interval,
		windowStart: time.Now(),
	}
}

func (r *FixedWindow) Allow() bool {
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
