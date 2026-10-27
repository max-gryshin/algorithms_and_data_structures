package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand/v2"
	"regexp"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"algorithms_and_data_structures/ai_engineering/evals"
	"algorithms_and_data_structures/ai_engineering/llm"
)

// session builds a tool-heavy history: task, then n × (tool_use, 2 KB tool_result).
func session(n int) []llm.Message {
	msgs := []llm.Message{{Role: llm.RoleUser, Content: "Найди секретный код."}}
	for i := 1; i <= n; i++ {
		id := fmt.Sprintf("t%d", i)
		msgs = append(msgs,
			llm.Message{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{call(id, "fetch_page", fmt.Sprintf(`{"page":%d}`, i))}, Raw: json.RawMessage(`[]`)},
			llm.Message{Role: llm.RoleUser, ToolResults: []llm.ToolResult{{CallID: id, Content: page(i)}}})
	}
	return msgs
}

func page(i int) string {
	body := fmt.Sprintf("PAGE %d\n", i)
	if i == 1 {
		body += "CODE=7351\n"
	}
	return body + strings.Repeat("filler text. ", 160) // ≈ 2 KB ≈ 520 tokens
}

func TestContextManagerUnderBudgetIsNoop(t *testing.T) {
	m := &ContextManager{MaxTokens: 100_000}
	msgs := session(3)
	out, rep, err := m.Fit(context.Background(), llm.Request{Messages: msgs})
	if err != nil || &out[0] != &msgs[0] || rep.Cleared != 0 || rep.Before != rep.After {
		t.Fatalf("under budget nothing changes: %+v %v", rep, err)
	}
}

func TestContextManagerClearsOldestToolResults(t *testing.T) {
	msgs := session(6) // ≈ 6 × 520 = 3 100 tokens
	m := &ContextManager{MaxTokens: 2500, Target: 2000, KeepToolResults: 2}
	out, rep, err := m.Fit(context.Background(), llm.Request{Messages: msgs})
	if err != nil {
		t.Fatal(err)
	}
	// Each cleared result saves ≈ 510 tokens: 3 100 → 2 000 needs 3 clears (pages 1-3).
	if rep.Cleared != 3 || rep.After > 2000 || rep.Before <= 2500 {
		t.Fatalf("report = %+v", rep)
	}
	for i := 1; i <= 6; i++ {
		got := out[2*i].ToolResults[0].Content
		if cleared := strings.HasPrefix(got, clearedPrefix); cleared != (i <= 3) {
			t.Errorf("page %d cleared = %v", i, cleared)
		}
	}
	if !strings.HasPrefix(msgs[2].ToolResults[0].Content, "PAGE 1") {
		t.Fatal("Fit must not modify the caller's messages")
	}
	if out[1].Raw != nil || msgs[1].Raw == nil {
		t.Fatal("after an edit thinking blocks (Raw) are not replayed")
	}
	if len(out) != len(msgs) || out[1].ToolCalls[0].ID != "t1" {
		t.Fatal("clearing keeps every tool_use/tool_result pair")
	}
}

// keepFacts is a fake summarizer: it keeps the lines with facts, like a good summary should.
var factLine = regexp.MustCompile(`(?m)^(CODE=\d+|PAGE \d+)$`)

func keepFacts() *fakeLLM {
	return &fakeLLM{fn: func(req llm.Request) (*llm.Response, error) {
		facts := factLine.FindAllString(req.Messages[0].Content, -1)
		return &llm.Response{Text: strings.Join(slices.Compact(facts), "\n"), Usage: llm.Usage{InputTokens: 500, OutputTokens: 40}}, nil
	}}
}

func TestContextManagerCompacts(t *testing.T) {
	msgs := session(8)
	sum := keepFacts()
	m := &ContextManager{MaxTokens: 2500, Target: 1500, KeepToolResults: 8, KeepRecent: 2, Summarizer: sum, Model: "small"}
	out, rep, err := m.Fit(context.Background(), llm.Request{Messages: msgs})
	if err != nil {
		t.Fatal(err)
	}
	// Cut at the last assistant message leaving ≥ 2 messages: index 15 (call for page 8).
	if rep.Compacted != 15 || len(out) != 3 || rep.Usage.OutputTokens != 40 {
		t.Fatalf("report = %+v, len = %d", rep, len(out))
	}
	head := out[0]
	if head.Role != llm.RoleUser || !strings.Contains(head.Content, "<task>\nНайди секретный код.\n</task>") || !strings.Contains(head.Content, "CODE=7351") {
		t.Fatalf("head = %q", head.Content)
	}
	if out[1].Role != llm.RoleAssistant || out[1].ToolCalls[0].ID != "t8" || out[2].ToolResults[0].CallID != "t8" {
		t.Fatal("the kept tail must start with an assistant turn and keep the tool_use/tool_result pair")
	}
	if got := sum.requests()[0]; got.Model != "small" || !strings.Contains(got.Messages[0].Content, "<transcript>") {
		t.Fatalf("summarizer request = %+v", got)
	}
	// A second compaction must keep the original task, not nest summaries.
	more := append(out, session(8)[1:]...)
	out2, _, err := m.Fit(context.Background(), llm.Request{Messages: more})
	if err != nil || strings.Count(out2[0].Content, "<task>") != 1 || !strings.Contains(out2[0].Content, "Найди секретный код.") {
		t.Fatalf("re-compaction head = %q, %v", out2[0].Content, err)
	}
}

func TestContextManagerOverflow(t *testing.T) {
	m := &ContextManager{MaxTokens: 100}
	_, _, err := m.Fit(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: strings.Repeat("x", 4000)}}})
	if !errors.Is(err, ErrContextOverflow) {
		t.Fatalf("err = %v", err)
	}
}

// pagerModel is a fake agent model with a context window: it reads pages 1..pages with
// fetch_page, then answers with the CODE it can still see in its prompt. A prompt over
// window tokens fails like a real API ("prompt is too long").
func pagerModel(pages, window int) *fakeLLM {
	pageRe := regexp.MustCompile(`PAGE (\d+)`)
	codeRe := regexp.MustCompile(`CODE=(\d+)`)
	return &fakeLLM{fn: func(req llm.Request) (*llm.Response, error) {
		if n := int(llm.EstimateTokens(llm.Request{System: req.System, Messages: req.Messages})); n > window {
			return nil, &llm.APIError{StatusCode: 400, Type: "invalid_request_error", Message: fmt.Sprintf("prompt is too long: %d tokens > %d", n, window)}
		}
		prompt := Transcript(req.Messages, 1<<20)
		seen := 0
		for _, m := range pageRe.FindAllStringSubmatch(prompt, -1) {
			var p int
			fmt.Sscan(m[1], &p)
			seen = max(seen, p)
		}
		usage := llm.Usage{InputTokens: int(llm.EstimateTokens(req)) - req.MaxTokens, OutputTokens: 30}
		if seen < pages {
			in := fmt.Sprintf(`{"page":%d}`, seen+1)
			return &llm.Response{StopReason: "tool_use", Usage: usage, ToolCalls: []llm.ToolCall{call(fmt.Sprintf("p%d", seen+1), "fetch_page", in)}}, nil
		}
		if m := codeRe.FindStringSubmatch(prompt); m != nil {
			return &llm.Response{StopReason: "end_turn", Text: "Код: " + m[1], Usage: usage}, nil
		}
		return &llm.Response{StopReason: "end_turn", Text: "Код не найден", Usage: usage}, nil
	}}
}

type pageInput struct {
	Page int `json:"page"`
}

func fetchPage() Tool {
	return NewTool("fetch_page", "Fetch a page of the document.", func(_ context.Context, in pageInput) (string, error) {
		return page(in.Page), nil
	})
}

// TestCompareContextStrategies runs the same dataset through three agents and compares
// them with evals.Runner and evals.Diff — the workflow of L20.
func TestCompareContextStrategies(t *testing.T) {
	ds := &evals.Dataset{Name: "long-sessions"}
	for _, pages := range []int{2, 4, 8, 12} {
		ds.Cases = append(ds.Cases, evals.Case{ID: fmt.Sprintf("pages=%d", pages), Input: fmt.Sprint(pages), Expected: "7351"})
	}
	const window = 4000 // tokens; 12 pages × 520 ≈ 6 200 do not fit
	strategy := func(cm func() *ContextManager) evals.Target {
		return func(ctx context.Context, input string) (string, error) {
			var pages int
			fmt.Sscan(input, &pages)
			a := &Agent{Client: pagerModel(pages, window), Tools: NewRegistry(fetchPage()), MaxSteps: 30, MaxTokens: 200}
			if cm != nil {
				a.Context = cm()
			}
			res, err := a.Run(ctx, "Прочитай все страницы и назови секретный код.")
			if err != nil {
				return "", err
			}
			return res.Text, nil
		}
	}
	runs := map[string]*evals.Run{}
	for name, cm := range map[string]func() *ContextManager{
		"none":    nil,
		"clear":   func() *ContextManager { return &ContextManager{MaxTokens: 3000, Target: 2000} },
		"compact": func() *ContextManager { return &ContextManager{MaxTokens: 3000, Target: 2000, Summarizer: keepFacts()} },
	} {
		r := &evals.Runner{Target: strategy(cm), Grader: evals.Contains{}, Workers: 2}
		run, err := r.Run(context.Background(), name, ds)
		if err != nil {
			t.Fatal(err)
		}
		runs[name] = run
	}
	acc := func(name string) float64 { return runs[name].Report(rand.New(rand.NewPCG(1, 2)), 200).Accuracy }
	// none:    long sessions overflow the window → API errors on 8 and 12 pages.
	// clear:   fits, but page 1 (with the code) is cleared before the answer.
	// compact: the summary keeps CODE=7351.
	if acc("none") != 0.5 || acc("clear") != 0.5 || acc("compact") != 1 {
		t.Fatalf("accuracy none=%.2f clear=%.2f compact=%.2f", acc("none"), acc("clear"), acc("compact"))
	}
	d := evals.Diff(runs["none"], runs["compact"])
	if !slices.Equal(d.Fixed, []string{"pages=8", "pages=12"}) || len(d.Broken) != 0 || d.BothPass != 2 {
		t.Fatalf("diff = %+v", d)
	}
	t.Logf("none → compact: fixed %v, broken %v, p = %.3f", d.Fixed, d.Broken, d.PValue)
}

func TestGroupLimitAndCancel(t *testing.T) {
	g, ctx := WithContext(context.Background())
	g.SetLimit(3)
	var running, peak atomic.Int32
	for range 10 {
		g.Go(func() error {
			n := running.Add(1)
			for {
				p := peak.Load()
				if n <= p || peak.CompareAndSwap(p, n) {
					break
				}
			}
			time.Sleep(5 * time.Millisecond)
			running.Add(-1)
			return nil
		})
	}
	if err := g.Wait(); err != nil || peak.Load() > 3 || peak.Load() < 2 {
		t.Fatalf("err=%v peak=%d", err, peak.Load())
	}
	if ctx.Err() == nil {
		t.Fatal("context is canceled after Wait")
	}

	boom := errors.New("boom")
	g, ctx = WithContext(context.Background())
	g.Go(func() error { return boom })
	g.Go(func() error { <-ctx.Done(); return ctx.Err() })
	if err := g.Wait(); !errors.Is(err, boom) || !errors.Is(context.Cause(ctx), boom) {
		t.Fatalf("first error wins and cancels the rest: %v", err)
	}
}

func TestOrchestratorWorkers(t *testing.T) {
	var inFlight, peak atomic.Int32
	orch := &fakeLLM{fn: func(req llm.Request) (*llm.Response, error) {
		if req.JSONSchema != nil { // planning call
			return &llm.Response{Text: `{"subtasks":["пакет rag","пакет llm","пакет evals","пакет agent"]}`, Usage: llm.Usage{InputTokens: 100, OutputTokens: 20}}, nil
		}
		return &llm.Response{Text: "СИНТЕЗ:\n" + req.Messages[0].Content, Usage: llm.Usage{InputTokens: 300, OutputTokens: 50}}, nil
	}}
	o := &Orchestrator{
		Client: orch, Concurrency: 2, MaxSubtasks: 3, Tracer: &evals.Tracer{},
		NewWorker: func(sub string) *Agent {
			w := &fakeLLM{fn: func(req llm.Request) (*llm.Response, error) {
				n := inFlight.Add(1)
				for p := peak.Load(); n > p && !peak.CompareAndSwap(p, n); p = peak.Load() {
				}
				time.Sleep(10 * time.Millisecond)
				inFlight.Add(-1)
				if sub == "пакет llm" {
					return nil, errors.New("worker crashed")
				}
				return &llm.Response{Text: "отчёт: " + req.Messages[0].Content, StopReason: "end_turn", Usage: llm.Usage{InputTokens: 1000, OutputTokens: 100}}, nil
			}}
			return &Agent{Client: w}
		},
	}
	res, err := o.Run(context.Background(), "Сделай обзор пакетов")
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Plan) != 3 || len(res.Workers) != 3 || peak.Load() != 2 {
		t.Fatalf("plan=%v peak=%d", res.Plan, peak.Load())
	}
	if res.Workers[1].Err == nil || res.Workers[0].Output != "отчёт: пакет rag" {
		t.Fatalf("workers = %+v", res.Workers)
	}
	for _, want := range []string{`subtask="пакет rag"`, "отчёт: пакет evals", "ОШИБКА: worker crashed"} {
		if !strings.Contains(res.Text, want) {
			t.Errorf("synthesis input lacks %q", want)
		}
	}
	// plan 120 + 2 workers × 1100 + synthesis 350 = 2 670
	if got := TotalTokens(res.Usage); got != 2670 {
		t.Fatalf("total tokens = %d", got)
	}

	o.FailFast = true
	if _, err := o.Run(context.Background(), "x"); err == nil || !strings.Contains(err.Error(), "worker crashed") {
		t.Fatalf("fail-fast: %v", err)
	}
}

// lostInTheMiddle is a fake model that only "sees" the first and last 25 % of a long
// document — a caricature of the U-shaped recall curve from Liu et al.
func lostInTheMiddle(needle, answer string, shortLimit int) *fakeLLM {
	return &fakeLLM{fn: func(req llm.Request) (*llm.Response, error) {
		doc := req.Messages[0].Content
		i := strings.Index(doc, needle)
		pos := float64(i) / float64(len(doc))
		if i >= 0 && (len(doc) < shortLimit*4 || pos < 0.25 || pos > 0.75) {
			return &llm.Response{Text: "Ответ: " + answer}, nil
		}
		return &llm.Response{Text: "В документе этого нет."}, nil
	}}
}

func TestNeedleInAHaystack(t *testing.T) {
	n := Needle{
		Filler:   []string{"Небо было серым.", "Поезд опаздывал на десять минут.", "В городе открыли новую библиотеку.", "Кофе остывал на столе."},
		Needle:   "Пароль от сервера — «гранат-42».",
		Question: "Какой пароль от сервера? Ответь одним словом.",
		Answer:   "гранат-42",
		Lengths:  []int{500, 4000},
		Depths:   []float64{0, 0.5, 1},
		Seed:     1,
	}
	ds := n.Dataset()
	if len(ds.Cases) != 6 || ds.Cases[1].ID != "len=500/depth=0.50" {
		t.Fatalf("cases = %d, %q", len(ds.Cases), ds.Cases[1].ID)
	}
	if got := len(ds.Cases[5].Input); got < 4000*4 || got > 4000*4+400 {
		t.Fatalf("haystack size = %d bytes, want ≈ 16 000", got)
	}
	model := lostInTheMiddle(n.Needle, n.Answer, 1000)
	r := &evals.Runner{Target: LLMTarget(model, "m", "", 64), Grader: evals.Contains{}, Workers: 3}
	run, err := r.Run(context.Background(), "needle", ds)
	if err != nil {
		t.Fatal(err)
	}
	grid := NeedleGrid(run)
	want := "              len=500  len=4000\n" +
		"depth=0.00          +         +\n" +
		"depth=0.50          +         -\n" +
		"depth=1.00          +         +\n"
	if grid != want {
		t.Fatalf("grid:\n%s\nwant:\n%s", grid, want)
	}
	rep := run.Report(rand.New(rand.NewPCG(1, 1)), 100)
	if rep.ByTag["depth=0.50"].Passed != 1 || rep.ByTag["len=500"].Accuracy != 1 {
		t.Fatalf("by tag = %+v", rep.ByTag)
	}
}
