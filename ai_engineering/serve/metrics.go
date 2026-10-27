package serve

import (
	"fmt"
	"io"
	"math"
	"slices"
	"sync"
	"time"
)

// Metrics collects serving latencies: TTFT (time to first token), ITL (inter-token
// latency, its mean per request is TPOT) and end-to-end latency. Safe for concurrent use.
type Metrics struct {
	mu        sync.Mutex
	start     time.Time
	ttft, itl []float64 // seconds
	e2e       []float64
	requests  int
	tokens    int
}

func NewMetrics() *Metrics { return &Metrics{start: time.Now()} }

func (m *Metrics) ObserveTTFT(d time.Duration) { m.add(&m.ttft, d) }
func (m *Metrics) ObserveITL(d time.Duration)  { m.add(&m.itl, d) }

// ObserveRequest records a finished request with its generated tokens.
func (m *Metrics) ObserveRequest(d time.Duration, tokens int) {
	m.mu.Lock()
	m.e2e = append(m.e2e, d.Seconds())
	m.requests++
	m.tokens += tokens
	m.mu.Unlock()
}

func (m *Metrics) add(xs *[]float64, d time.Duration) {
	m.mu.Lock()
	*xs = append(*xs, d.Seconds())
	m.mu.Unlock()
}

// Percentile is the linearly interpolated p-quantile (0 ≤ p ≤ 1) of xs:
// rank r = p·(n−1), between sorted xs[⌊r⌋] and xs[⌈r⌉]. NaN for empty xs.
func Percentile(xs []float64, p float64) float64 {
	if len(xs) == 0 {
		return math.NaN()
	}
	s := slices.Clone(xs)
	slices.Sort(s)
	r := p * float64(len(s)-1)
	lo := int(math.Floor(r))
	hi := min(lo+1, len(s)-1)
	return s[lo] + (r-float64(lo))*(s[hi]-s[lo])
}

// Summary is a snapshot of the metrics.
type Summary struct {
	Requests, Tokens          int
	TTFTp50, TTFTp95, TTFTp99 float64 // seconds
	ITLp50, ITLp95            float64
	E2Ep50                    float64
	Throughput                float64 // generated tokens per second since start
}

func (m *Metrics) Summary() Summary {
	m.mu.Lock()
	defer m.mu.Unlock()
	el := time.Since(m.start).Seconds()
	return Summary{Requests: m.requests, Tokens: m.tokens,
		TTFTp50: Percentile(m.ttft, .5), TTFTp95: Percentile(m.ttft, .95), TTFTp99: Percentile(m.ttft, .99),
		ITLp50: Percentile(m.itl, .5), ITLp95: Percentile(m.itl, .95), E2Ep50: Percentile(m.e2e, .5),
		Throughput: float64(m.tokens) / el}
}

// WriteText writes the summary in the Prometheus text format.
func (m *Metrics) WriteText(w io.Writer) {
	s := m.Summary()
	fmt.Fprintf(w, "serve_requests_total %d\nserve_generated_tokens_total %d\n", s.Requests, s.Tokens)
	fmt.Fprintf(w, "serve_ttft_seconds{quantile=\"0.5\"} %g\nserve_ttft_seconds{quantile=\"0.95\"} %g\n", s.TTFTp50, s.TTFTp95)
	fmt.Fprintf(w, "serve_itl_seconds{quantile=\"0.5\"} %g\nserve_itl_seconds{quantile=\"0.95\"} %g\n", s.ITLp50, s.ITLp95)
	fmt.Fprintf(w, "serve_throughput_tokens_per_second %g\n", s.Throughput)
}
