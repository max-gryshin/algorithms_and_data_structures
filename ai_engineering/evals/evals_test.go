package evals

import (
	"bytes"
	"context"
	"errors"
	"math"
	"math/rand/v2"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func near(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

func TestLoadDatasetFormats(t *testing.T) {
	obj := `{"name":"qa","cases":[{"id":"1","input":"2+2?","expected":"4","tags":["math"]}]}`
	ds, err := LoadDataset(strings.NewReader(obj))
	if err != nil || ds.Name != "qa" || len(ds.Cases) != 1 || ds.Cases[0].Tags[0] != "math" {
		t.Fatalf("object: %+v %v", ds, err)
	}
	jsonl := "{\"id\":\"a\",\"input\":\"x\"}\n\n{\"id\":\"b\",\"input\":\"y\"}\n"
	ds, err = LoadDataset(strings.NewReader(jsonl))
	if err != nil || len(ds.Cases) != 2 || ds.Cases[1].ID != "b" {
		t.Fatalf("jsonl: %+v %v", ds, err)
	}
	if _, err := LoadDataset(strings.NewReader("{\"id\":\"a\"}\n{\"id\":\"a\"}")); err == nil {
		t.Fatal("duplicate ids must fail")
	}
}

func grade(t *testing.T, g Grader, c Case, out string) Grade {
	t.Helper()
	gr, err := g.Grade(context.Background(), c, out)
	if err != nil {
		t.Fatal(err)
	}
	return gr
}

func TestGraders(t *testing.T) {
	c := Case{ID: "1", Expected: "Париж"}
	if !grade(t, Exact{IgnoreCase: true, TrimSpace: true}, c, "  париж\n").Pass {
		t.Fatal("exact")
	}
	if grade(t, Exact{}, c, "париж").Pass {
		t.Fatal("exact is case-sensitive by default")
	}
	if !grade(t, Contains{}, c, "Столица Франции — ПАРИЖ.").Pass {
		t.Fatal("contains")
	}
	if !grade(t, Regex{}, Case{Expected: `^\d{4}$`}, "2026").Pass {
		t.Fatal("regex")
	}
	shape := JSONShape{Required: map[string]string{"city": "string", "population": "number"}}
	if !grade(t, shape, c, "```json\n{\"city\":\"Paris\",\"population\":2.1e6}\n```").Pass {
		t.Fatal("json shape")
	}
	if gr := grade(t, shape, c, `{"city":"Paris","population":"2M"}`); gr.Pass || !strings.Contains(gr.Reason, "population") {
		t.Fatalf("json shape must fail on wrong type: %+v", gr)
	}
	if gr := grade(t, All(Contains{}, shape), c, `{"city":"Париж","population":2}`); !gr.Pass || gr.Score != 1 {
		t.Fatalf("all: %+v", gr)
	}
}

type fakeModel func(prompt string) string

func (f fakeModel) Generate(_ context.Context, p string) (string, error) { return f(p), nil }

func TestLLMJudge(t *testing.T) {
	judge := LLMJudge{
		Rubric: "Ответ должен назвать столицу.",
		Model: fakeModel(func(p string) string {
			if strings.Contains(p, "<answer>\nПариж") {
				return "Назван Париж.\nVERDICT: PASS"
			}
			return "Столица не названа.\nverdict: fail"
		}),
	}
	c := Case{ID: "1", Input: "Столица Франции?"}
	if gr := grade(t, judge, c, "Париж"); !gr.Pass || gr.Reason != "Назван Париж." {
		t.Fatalf("%+v", gr)
	}
	if gr := grade(t, judge, c, "Не знаю"); gr.Pass {
		t.Fatalf("%+v", gr)
	}
	broken := LLMJudge{Model: fakeModel(func(string) string { return "хороший ответ" })}
	if _, err := broken.Grade(context.Background(), c, "x"); err == nil {
		t.Fatal("reply without verdict must be an error")
	}
}

func TestPairwiseCancelsPositionBias(t *testing.T) {
	ctx := context.Background()
	alwaysFirst := fakeModel(func(string) string { return "VERDICT: 1" })
	if w, _ := Pairwise(ctx, alwaysFirst, "q", "A", "B"); w != 0 {
		t.Fatalf("a judge that always picks position 1 must yield a tie, got %d", w)
	}
	prefersLong := fakeModel(func(p string) string {
		a1 := p[strings.Index(p, "Ответ 1:"):strings.Index(p, "Ответ 2:")]
		if strings.Contains(a1, "подробный") {
			return "VERDICT: 1"
		}
		return "VERDICT: 2"
	})
	if w, _ := Pairwise(ctx, prefersLong, "q", "короткий", "подробный"); w != -1 {
		t.Fatalf("B must win in both orders, got %d", w)
	}
}

func TestRunnerConcurrentAndErrors(t *testing.T) {
	var inflight, peak atomic.Int32
	target := func(ctx context.Context, in string) (string, error) {
		n := inflight.Add(1)
		defer inflight.Add(-1)
		for {
			p := peak.Load()
			if n <= p || peak.CompareAndSwap(p, n) {
				break
			}
		}
		time.Sleep(5 * time.Millisecond)
		if in == "boom" {
			return "", errors.New("api 500")
		}
		return strings.ToUpper(in), nil
	}
	ds := &Dataset{Cases: []Case{
		{ID: "1", Input: "a", Expected: "A", Tags: []string{"easy"}},
		{ID: "2", Input: "b", Expected: "B", Tags: []string{"easy"}},
		{ID: "3", Input: "c", Expected: "x", Tags: []string{"hard"}},
		{ID: "4", Input: "boom", Expected: "BOOM", Tags: []string{"hard"}},
		{ID: "5", Input: "e", Expected: "E"},
		{ID: "6", Input: "f", Expected: "F"},
	}}
	r := &Runner{Target: target, Grader: Exact{}, Workers: 3}
	run, err := r.Run(context.Background(), "v1", ds)
	if err != nil {
		t.Fatal(err)
	}
	if peak.Load() > 3 {
		t.Fatalf("more than 3 workers in flight: %d", peak.Load())
	}
	for i, res := range run.Results {
		if res.Case.ID != ds.Cases[i].ID {
			t.Fatal("results must keep dataset order")
		}
	}
	rep := run.Report(rand.New(rand.NewPCG(1, 1)), 2000)
	if rep.Passed != 4 || rep.Errors != 1 || !near(rep.Accuracy, 4.0/6, 1e-12) {
		t.Fatalf("report: %+v", rep)
	}
	if rep.ByTag["easy"].Accuracy != 1 || rep.ByTag["hard"].Accuracy != 0 {
		t.Fatalf("by tag: %+v", rep.ByTag)
	}
	if !(rep.CILow < rep.Accuracy && rep.Accuracy < rep.CIHigh) || rep.P50 < 5*time.Millisecond {
		t.Fatalf("ci/latency: %+v", rep)
	}
	t.Log("\n" + rep.String())
}

func TestRunnerCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	r := &Runner{Target: func(context.Context, string) (string, error) { return "", nil }, Grader: Exact{}}
	if _, err := r.Run(ctx, "x", &Dataset{Cases: []Case{{ID: "1"}, {ID: "2"}}}); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v", err)
	}
}

// Hand-computed examples from lecture L19.
func TestStatsHandExamples(t *testing.T) {
	xs := []float64{1, 0, 1, 1, 0}
	if Mean(xs) != 0.6 || !near(StdErr(xs), 0.2449, 1e-4) {
		t.Fatalf("mean=%v se=%v", Mean(xs), StdErr(xs))
	}
	lat := []float64{120, 95, 300, 110, 105, 2000, 130, 98, 115, 140}
	if p := Percentile(lat, 50); !near(p, 117.5, 1e-9) {
		t.Fatalf("p50 = %v", p)
	}
	if p := Percentile(lat, 95); !near(p, 1235, 1e-6) {
		t.Fatalf("p95 = %v", p)
	}
	if p := Percentile(lat, 99); !near(p, 1847, 1e-6) {
		t.Fatalf("p99 = %v", p)
	}
	if p := McNemar(15, 5); !near(p, 0.04139, 1e-5) {
		t.Fatalf("mcnemar = %v", p)
	}
	if McNemar(0, 0) != 1 || McNemar(3, 3) != 1 {
		t.Fatal("no difference must give p = 1")
	}
	if n := SampleSize(0.5, 0.03); n != 1068 {
		t.Fatalf("sample size = %d", n)
	}
}

func TestBootstrapMatchesNormal(t *testing.T) {
	// 80 passes out of 100: textbook CI is 0.8 ± 1.96·0.04 = [0.722, 0.878].
	xs := make([]float64, 100)
	for i := range 80 {
		xs[i] = 1
	}
	lo, hi := Bootstrap(xs, 10000, 0.05, rand.New(rand.NewPCG(42, 42)))
	if !near(lo, 0.72, 0.015) || !near(hi, 0.88, 0.015) {
		t.Fatalf("bootstrap CI = [%.3f, %.3f]", lo, hi)
	}
	lo2, hi2 := Bootstrap(xs, 10000, 0.05, rand.New(rand.NewPCG(42, 42)))
	if lo != lo2 || hi != hi2 {
		t.Fatal("same seed must give the same interval")
	}
}

func TestDiff(t *testing.T) {
	mk := func(name string, pass ...bool) *Run {
		r := &Run{Name: name}
		for i, p := range pass {
			r.Results = append(r.Results, Result{Case: Case{ID: string(rune('a' + i))}, Grade: Grade{Pass: p}})
		}
		return r
	}
	a := mk("old", true, true, false, false, true)
	b := mk("new", true, false, true, true, true)
	d := Diff(a, b)
	if d.BothPass != 2 || d.BothFail != 0 || len(d.Fixed) != 2 || len(d.Broken) != 1 || d.Broken[0] != "b" {
		t.Fatalf("%+v", d)
	}
	if d.PValue != 1 { // 2 vs 1 discordant: no evidence of a difference
		t.Fatalf("p = %v", d.PValue)
	}
}

func TestTracer(t *testing.T) {
	tr := &Tracer{}
	ctx, root := tr.Start(context.Background(), "rag.answer")
	_, child := tr.Start(ctx, "llm.generate")
	child.SetAttr("model", "fake")
	child.SetAttr("output_tokens", 42)
	child.Finish(nil)
	root.Finish(errors.New("timeout"))

	spans := tr.Spans()
	if len(spans) != 2 || spans[0].ParentID != root.SpanID || spans[0].TraceID != root.TraceID {
		t.Fatalf("bad tree: %+v", spans)
	}
	if root.Error != "timeout" || root.Duration() < child.Duration() {
		t.Fatal("root must hold the error and outlast its child")
	}
	var buf bytes.Buffer
	if err := tr.ExportJSON(&buf); err != nil {
		t.Fatal(err)
	}
	if lines := strings.Count(buf.String(), "\n"); lines != 2 || !strings.Contains(buf.String(), `"output_tokens":42`) {
		t.Fatalf("export: %s", buf.String())
	}
}
