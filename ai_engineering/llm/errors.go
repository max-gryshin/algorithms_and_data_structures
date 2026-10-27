package llm

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"time"
)

// APIError is a non-2xx HTTP answer (or an error event inside a stream).
type APIError struct {
	StatusCode int
	Type       string // e.g. rate_limit_error, overloaded_error, invalid_request_error
	Message    string
	RetryAfter time.Duration // from the retry-after header, 0 if absent
}

func (e *APIError) Error() string {
	return fmt.Sprintf("llm: HTTP %d %s: %s", e.StatusCode, e.Type, e.Message)
}

// Retryable reports whether repeating the same request may succeed:
// 408 timeout, 409 conflict, 429 rate limit, 5xx server errors and 529 overloaded.
// 4xx errors such as 400 or 401 will fail again and must not be retried.
func (e *APIError) Retryable() bool {
	switch {
	case e.StatusCode == http.StatusRequestTimeout, e.StatusCode == http.StatusConflict,
		e.StatusCode == http.StatusTooManyRequests, e.StatusCode >= 500:
		return true
	}
	return false
}

// IsRetryable classifies any error returned by a Client.
func IsRetryable(err error) bool {
	if err == nil || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	if apiErr, ok := errors.AsType[*APIError](err); ok {
		return apiErr.Retryable()
	}
	if errors.Is(err, ErrCircuitOpen) || errors.Is(err, ErrBudgetExceeded) {
		return false
	}
	return true // transport errors: connection reset, EOF in the middle of a body, DNS hiccups
}

// parseRetryAfter reads a retry-after header given in seconds.
func parseRetryAfter(h http.Header) time.Duration {
	if s, err := strconv.ParseFloat(h.Get("retry-after"), 64); err == nil && s > 0 {
		return time.Duration(s * float64(time.Second))
	}
	return 0
}
