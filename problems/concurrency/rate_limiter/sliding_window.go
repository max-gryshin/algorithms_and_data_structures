package rate_limiter

import (
	"sync"
	"time"
)

type SlidingWindow struct {
	mx              sync.Mutex
	interval        time.Duration
	windowStart     time.Time
	prevWindowStart time.Time
	count           int
	prevCount       int
	limit           int
}

func NewSlidingWindow(limit int, interval time.Duration) *SlidingWindow {
	return &SlidingWindow{
		interval:    interval,
		windowStart: time.Now(),
		limit:       limit,
	}
}

func (sw *SlidingWindow) Allow() bool {
	sw.mx.Lock()
	defer sw.mx.Unlock()
	now := time.Now()
	if now.Sub(sw.windowStart) >= sw.interval {
		sw.prevWindowStart = sw.windowStart
		sw.prevCount = sw.count
		sw.windowStart = now
		sw.count = 0
	}

	windowSeconds := now.Sub(sw.windowStart).Seconds()
	intervalSeconds := sw.interval.Seconds()
	weightPrev := (intervalSeconds - windowSeconds) / intervalSeconds

	weightedCount := weightPrev*float64(sw.prevCount) + float64(sw.count)
	if weightedCount >= float64(sw.limit) {
		return false
	}

	sw.count++
	return true
}
