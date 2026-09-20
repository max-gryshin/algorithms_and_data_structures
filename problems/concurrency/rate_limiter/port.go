package rate_limiter

type RateLimiter interface {
	Allow() bool
}
