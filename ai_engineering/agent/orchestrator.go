package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"algorithms_and_data_structures/ai_engineering/evals"
	"algorithms_and_data_structures/ai_engineering/llm"
)

// Orchestrator implements the orchestrator-workers pattern (L18, L20):
//
//	plan       — one LLM call splits the task into independent subtasks (structured output);
//	workers    — each subtask runs in its own Agent with a fresh context, concurrently;
//	synthesize — one LLM call merges the workers' condensed reports into the answer.
//
// The win is context isolation: each worker may read 100k tokens of files, but only its
// short report reaches the orchestrator. The cost is more tokens in total and the risk
// that subtasks are not really independent.
type Orchestrator struct {
	Client      llm.Client
	Model       string
	MaxSubtasks int // cap on the plan; default 5
	Concurrency int // workers running at once; default 3
	MaxTokens   int // for plan and synthesis calls; default 4096

	// NewWorker builds a fresh agent for one subtask (its own history = its own context).
	NewWorker func(subtask string) *Agent
	// FailFast cancels all workers on the first error; otherwise failures are reported
	// to the synthesizer as such and the rest of the work is used.
	FailFast bool
	// MaxReport is the number of bytes of each worker's answer passed to synthesis; default 4 KiB.
	MaxReport int
	Tracer    *evals.Tracer
}

// WorkerReport is the outcome of one subtask.
type WorkerReport struct {
	Subtask string
	Output  string
	Err     error
	Steps   int
	Usage   llm.Usage
}

// OrchestratorResult is the outcome of Run.
type OrchestratorResult struct {
	Plan    []string
	Workers []WorkerReport
	Text    string
	Usage   llm.Usage // orchestrator calls + all workers
}

var planSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"subtasks": map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
	},
	"required":             []string{"subtasks"},
	"additionalProperties": false,
}

const planSystem = `Ты — оркестратор. Разбей задачу на независимые подзадачи, которые можно выполнять параллельно,
не зная результатов друг друга. Каждая подзадача должна быть самодостаточной: исполнитель увидит только её текст.
Не дроби без необходимости: если задача простая, верни одну подзадачу.
Ответь JSON-объектом {"subtasks": ["...", "..."]}.`

// Run executes plan → workers → synthesis.
func (o *Orchestrator) Run(ctx context.Context, task string) (*OrchestratorResult, error) {
	ctx, end := startSpan(o.Tracer, ctx, "orchestrator.run")
	res := &OrchestratorResult{}
	var err error
	defer func() { end(err, "subtasks", len(res.Plan), "total_tokens", TotalTokens(res.Usage)) }()

	res.Plan, err = o.plan(ctx, task, res)
	if err != nil {
		return res, err
	}
	res.Workers = make([]WorkerReport, len(res.Plan))
	if err = o.runWorkers(ctx, res); err != nil {
		return res, err
	}
	err = o.synthesize(ctx, task, res)
	return res, err
}

func (o *Orchestrator) maxTokens() int {
	if o.MaxTokens > 0 {
		return o.MaxTokens
	}
	return 4096
}

func (o *Orchestrator) plan(ctx context.Context, task string, res *OrchestratorResult) ([]string, error) {
	ctx, end := startSpan(o.Tracer, ctx, "orchestrator.plan")
	resp, err := o.Client.Complete(ctx, llm.Request{
		Model: o.Model, System: planSystem, MaxTokens: o.maxTokens(), JSONSchema: planSchema,
		Messages: []llm.Message{{Role: llm.RoleUser, Content: task}},
	})
	if err != nil {
		end(err)
		return nil, err
	}
	res.Usage = res.Usage.Add(resp.Usage)
	var plan struct {
		Subtasks []string `json:"subtasks"`
	}
	if err := json.Unmarshal([]byte(resp.Text), &plan); err != nil || len(plan.Subtasks) == 0 {
		err = fmt.Errorf("agent: bad plan %q: %v", resp.Text, err)
		end(err)
		return nil, err
	}
	limit := o.MaxSubtasks
	if limit <= 0 {
		limit = 5
	}
	if len(plan.Subtasks) > limit {
		plan.Subtasks = plan.Subtasks[:limit]
	}
	end(nil, "subtasks", len(plan.Subtasks))
	return plan.Subtasks, nil
}

// runWorkers fans out with a bounded Group. Each goroutine writes only its own slot of
// res.Workers, so no mutex is needed; usage is summed after Wait.
func (o *Orchestrator) runWorkers(ctx context.Context, res *OrchestratorResult) error {
	g, gctx := WithContext(ctx)
	conc := o.Concurrency
	if conc <= 0 {
		conc = 3
	}
	g.SetLimit(conc)
	for i, sub := range res.Plan {
		g.Go(func() error {
			wctx, end := startSpan(o.Tracer, gctx, "orchestrator.worker")
			w := o.NewWorker(sub)
			if w.Tracer == nil {
				w.Tracer = o.Tracer
			}
			r, err := w.Run(wctx, sub)
			rep := WorkerReport{Subtask: sub, Err: err}
			if r != nil {
				rep.Output, rep.Steps, rep.Usage = r.Text, r.Steps, r.Usage
			}
			res.Workers[i] = rep
			end(err, "index", i, "steps", rep.Steps)
			if o.FailFast && err != nil {
				return fmt.Errorf("worker %d (%q): %w", i, sub, err)
			}
			return nil
		})
	}
	err := g.Wait()
	for _, w := range res.Workers {
		res.Usage = res.Usage.Add(w.Usage)
	}
	return err
}

func (o *Orchestrator) synthesize(ctx context.Context, task string, res *OrchestratorResult) error {
	ctx, end := startSpan(o.Tracer, ctx, "orchestrator.synthesize")
	limit := o.MaxReport
	if limit <= 0 {
		limit = 4 << 10
	}
	var b strings.Builder
	fmt.Fprintf(&b, "<task>\n%s\n</task>\n\n", task)
	for i, w := range res.Workers {
		out := truncate(w.Output, limit)
		if w.Err != nil {
			out = fmt.Sprintf("ОШИБКА: %v\n%s", w.Err, out)
		}
		fmt.Fprintf(&b, "<worker id=\"%d\" subtask=%q>\n%s\n</worker>\n", i+1, w.Subtask, out)
	}
	b.WriteString("\nСобери из отчётов исполнителей итоговый ответ на задачу. Отметь, какие части не удалось выполнить.")
	resp, err := o.Client.Complete(ctx, llm.Request{
		Model: o.Model, MaxTokens: o.maxTokens(),
		Messages: []llm.Message{{Role: llm.RoleUser, Content: b.String()}},
	})
	end(err)
	if err != nil {
		return err
	}
	res.Usage = res.Usage.Add(resp.Usage)
	res.Text = resp.Text
	return nil
}
