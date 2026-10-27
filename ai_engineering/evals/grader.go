package evals

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

// Grade is the verdict for one output.
type Grade struct {
	Pass   bool    `json:"pass"`
	Score  float64 `json:"score"` // 0..1; 1 or 0 for binary graders
	Reason string  `json:"reason,omitempty"`
}

func passed(ok bool, reason string) Grade {
	if ok {
		return Grade{Pass: true, Score: 1}
	}
	return Grade{Score: 0, Reason: reason}
}

// Grader checks one output against its case.
type Grader interface {
	Grade(ctx context.Context, c Case, output string) (Grade, error)
}

// GraderFunc adapts a function to Grader.
type GraderFunc func(ctx context.Context, c Case, output string) (Grade, error)

func (f GraderFunc) Grade(ctx context.Context, c Case, output string) (Grade, error) {
	return f(ctx, c, output)
}

// Generator is any LLM behind any API (used by LLMJudge).
type Generator interface {
	Generate(ctx context.Context, prompt string) (string, error)
}

// Exact passes when output equals c.Expected (after optional trimming / case folding).
type Exact struct{ IgnoreCase, TrimSpace bool }

func (e Exact) Grade(_ context.Context, c Case, out string) (Grade, error) {
	want := c.Expected
	if e.TrimSpace {
		out, want = strings.TrimSpace(out), strings.TrimSpace(want)
	}
	if e.IgnoreCase {
		out, want = strings.ToLower(out), strings.ToLower(want)
	}
	return passed(out == want, fmt.Sprintf("want %q, got %q", want, out)), nil
}

// Contains passes when the output contains c.Expected, case-insensitively.
type Contains struct{}

func (Contains) Grade(_ context.Context, c Case, out string) (Grade, error) {
	ok := strings.Contains(strings.ToLower(out), strings.ToLower(c.Expected))
	return passed(ok, fmt.Sprintf("output does not contain %q", c.Expected)), nil
}

// Regex passes when the output matches Pattern, or c.Expected compiled as a regexp if Pattern is nil.
type Regex struct{ Pattern *regexp.Regexp }

func (r Regex) Grade(_ context.Context, c Case, out string) (Grade, error) {
	re := r.Pattern
	if re == nil {
		var err error
		if re, err = regexp.Compile(c.Expected); err != nil {
			return Grade{}, fmt.Errorf("evals: case %s: bad regexp: %w", c.ID, err)
		}
	}
	return passed(re.MatchString(out), fmt.Sprintf("output does not match /%s/", re)), nil
}

// JSONShape is a tiny subset of JSON Schema: the output must be a JSON object with
// the Required keys of the given types ("string", "number", "bool", "array", "object", "null").
// A ```json fence around the object is tolerated, since models often add one.
type JSONShape struct{ Required map[string]string }

func (s JSONShape) Grade(_ context.Context, _ Case, out string) (Grade, error) {
	out = strings.TrimSpace(out)
	out = strings.TrimPrefix(out, "```json")
	out = strings.TrimSuffix(strings.TrimPrefix(out, "```"), "```")
	var obj map[string]any
	if err := json.Unmarshal([]byte(out), &obj); err != nil {
		return passed(false, "not a JSON object: "+err.Error()), nil
	}
	for key, typ := range s.Required {
		v, ok := obj[key]
		if !ok {
			return passed(false, "missing key "+key), nil
		}
		if got := jsonType(v); got != typ {
			return passed(false, fmt.Sprintf("key %s: want %s, got %s", key, typ, got)), nil
		}
	}
	return passed(true, ""), nil
}

func jsonType(v any) string {
	switch v.(type) {
	case string:
		return "string"
	case float64:
		return "number"
	case bool:
		return "bool"
	case []any:
		return "array"
	case map[string]any:
		return "object"
	default:
		return "null"
	}
}

// All passes only if every grader passes; Score is the mean score.
func All(graders ...Grader) Grader {
	return GraderFunc(func(ctx context.Context, c Case, out string) (Grade, error) {
		total := Grade{Pass: true}
		var reasons []string
		for _, g := range graders {
			gr, err := g.Grade(ctx, c, out)
			if err != nil {
				return Grade{}, err
			}
			total.Pass = total.Pass && gr.Pass
			total.Score += gr.Score / float64(len(graders))
			if gr.Reason != "" && !gr.Pass {
				reasons = append(reasons, gr.Reason)
			}
		}
		total.Reason = strings.Join(reasons, "; ")
		return total, nil
	})
}

// LLMJudge asks a model to grade the output against a rubric. The model must end its
// reply with a line "VERDICT: PASS" or "VERDICT: FAIL"; the text before it is the reason.
type LLMJudge struct {
	Model  Generator
	Rubric string // what a good answer looks like
}

func (j LLMJudge) Prompt(c Case, out string) string {
	var b strings.Builder
	b.WriteString("Ты — строгий проверяющий. Оцени ответ ассистента по критериям.\n\n")
	fmt.Fprintf(&b, "<criteria>\n%s\n</criteria>\n\n<question>\n%s\n</question>\n\n", j.Rubric, c.Input)
	if c.Expected != "" {
		fmt.Fprintf(&b, "<reference_answer>\n%s\n</reference_answer>\n\n", c.Expected)
	}
	fmt.Fprintf(&b, "<answer>\n%s\n</answer>\n\n", out)
	b.WriteString("Сначала кратко обоснуй оценку, затем последней строкой напиши ровно " +
		"\"VERDICT: PASS\" или \"VERDICT: FAIL\".")
	return b.String()
}

func (j LLMJudge) Grade(ctx context.Context, c Case, out string) (Grade, error) {
	reply, err := j.Model.Generate(ctx, j.Prompt(c, out))
	if err != nil {
		return Grade{}, fmt.Errorf("evals: judge: %w", err)
	}
	verdict, reason := parseVerdict(reply)
	switch verdict {
	case "PASS":
		return Grade{Pass: true, Score: 1, Reason: reason}, nil
	case "FAIL":
		return Grade{Score: 0, Reason: reason}, nil
	}
	return Grade{}, fmt.Errorf("evals: judge reply has no verdict: %q", reply)
}

func parseVerdict(reply string) (verdict, reason string) {
	lines := strings.Split(strings.TrimSpace(reply), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		l := strings.ToUpper(strings.TrimSpace(lines[i]))
		if v, ok := strings.CutPrefix(l, "VERDICT:"); ok {
			return strings.TrimSpace(v), strings.TrimSpace(strings.Join(lines[:i], "\n"))
		}
	}
	return "", ""
}

// Pairwise asks the judge which of two answers is better, twice — with A first and with
// B first — to cancel position bias. It returns +1 if A wins both times, −1 if B wins both,
// and 0 if the verdicts disagree (a tie, or the judge followed position rather than content).
func Pairwise(ctx context.Context, judge Generator, question, a, b string) (int, error) {
	ask := func(first, second string) (string, error) {
		p := fmt.Sprintf("Вопрос:\n%s\n\nОтвет 1:\n%s\n\nОтвет 2:\n%s\n\n"+
			"Какой ответ лучше? Последней строкой напиши ровно \"VERDICT: 1\" или \"VERDICT: 2\".",
			question, first, second)
		reply, err := judge.Generate(ctx, p)
		if err != nil {
			return "", err
		}
		v, _ := parseVerdict(reply)
		return v, nil
	}
	v1, err := ask(a, b)
	if err != nil {
		return 0, err
	}
	v2, err := ask(b, a)
	if err != nil {
		return 0, err
	}
	switch {
	case v1 == "1" && v2 == "2":
		return 1, nil
	case v1 == "2" && v2 == "1":
		return -1, nil
	}
	return 0, nil
}
