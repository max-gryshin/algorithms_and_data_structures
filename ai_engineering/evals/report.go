package evals

import (
	"fmt"
	"math/rand/v2"
	"sort"
	"strings"
	"time"
)

// TagStats is the pass rate for one tag.
type TagStats struct {
	N        int     `json:"n"`
	Passed   int     `json:"passed"`
	Accuracy float64 `json:"accuracy"`
}

// Report summarizes a run.
type Report struct {
	Name      string              `json:"name"`
	N         int                 `json:"n"`
	Passed    int                 `json:"passed"`
	Errors    int                 `json:"errors"`
	Accuracy  float64             `json:"accuracy"`
	CILow     float64             `json:"ci_low"` // 95 % bootstrap interval of accuracy
	CIHigh    float64             `json:"ci_high"`
	MeanScore float64             `json:"mean_score"`
	ByTag     map[string]TagStats `json:"by_tag"`
	P50       time.Duration       `json:"p50_ns"`
	P95       time.Duration       `json:"p95_ns"`
	P99       time.Duration       `json:"p99_ns"`
}

// Report computes accuracy with a 95 % bootstrap interval (iters resamples drawn from rng),
// per-tag accuracy and latency percentiles.
func (r *Run) Report(rng *rand.Rand, iters int) Report {
	rep := Report{Name: r.Name, N: len(r.Results), ByTag: map[string]TagStats{}}
	pass := make([]float64, len(r.Results))
	scores := make([]float64, len(r.Results))
	lat := make([]float64, len(r.Results))
	for i, res := range r.Results {
		ok := res.Err == "" && res.Grade.Pass
		if ok {
			rep.Passed++
			pass[i] = 1
		}
		if res.Err != "" {
			rep.Errors++
		}
		scores[i] = res.Grade.Score
		lat[i] = float64(res.Latency)
		for _, t := range res.Case.Tags {
			ts := rep.ByTag[t]
			ts.N++
			if ok {
				ts.Passed++
			}
			ts.Accuracy = float64(ts.Passed) / float64(ts.N)
			rep.ByTag[t] = ts
		}
	}
	rep.Accuracy = Mean(pass)
	rep.MeanScore = Mean(scores)
	rep.CILow, rep.CIHigh = Bootstrap(pass, iters, 0.05, rng)
	rep.P50 = time.Duration(Percentile(lat, 50))
	rep.P95 = time.Duration(Percentile(lat, 95))
	rep.P99 = time.Duration(Percentile(lat, 99))
	return rep
}

func (r Report) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s: %d/%d passed, accuracy %.1f%% (95%% CI %.1f–%.1f%%), errors %d\n",
		r.Name, r.Passed, r.N, 100*r.Accuracy, 100*r.CILow, 100*r.CIHigh, r.Errors)
	fmt.Fprintf(&b, "latency p50 %v, p95 %v, p99 %v\n", r.P50, r.P95, r.P99)
	tags := make([]string, 0, len(r.ByTag))
	for t := range r.ByTag {
		tags = append(tags, t)
	}
	sort.Strings(tags)
	for _, t := range tags {
		ts := r.ByTag[t]
		fmt.Fprintf(&b, "  %-12s %d/%d (%.0f%%)\n", t, ts.Passed, ts.N, 100*ts.Accuracy)
	}
	return b.String()
}

// DiffReport compares two runs over the same case IDs.
type DiffReport struct {
	BothPass, BothFail int
	Fixed              []string // failed in A, pass in B
	Broken             []string // passed in A, fail in B
	PValue             float64  // McNemar exact test on len(Broken) vs len(Fixed)
}

// Diff matches results by case ID; cases present in only one run are ignored.
func Diff(a, b *Run) DiffReport {
	inB := map[string]Result{}
	for _, r := range b.Results {
		inB[r.Case.ID] = r
	}
	var d DiffReport
	for _, ra := range a.Results {
		rb, ok := inB[ra.Case.ID]
		if !ok {
			continue
		}
		pa, pb := ra.Err == "" && ra.Grade.Pass, rb.Err == "" && rb.Grade.Pass
		switch {
		case pa && pb:
			d.BothPass++
		case !pa && !pb:
			d.BothFail++
		case pa:
			d.Broken = append(d.Broken, ra.Case.ID)
		default:
			d.Fixed = append(d.Fixed, ra.Case.ID)
		}
	}
	d.PValue = McNemar(len(d.Broken), len(d.Fixed))
	return d
}
