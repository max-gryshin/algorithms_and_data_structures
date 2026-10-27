package agent

import (
	"context"
	"fmt"
	"math/rand/v2"
	"slices"
	"strconv"
	"strings"

	"algorithms_and_data_structures/ai_engineering/evals"
	"algorithms_and_data_structures/ai_engineering/llm"
)

// Needle describes a needle-in-a-haystack experiment: a fact (the needle) is hidden at a
// relative depth inside filler text of a given length, and the model is asked about it.
// The result is a grid "context length × depth → found?". Dataset builds it as an
// evals.Dataset, so the usual Runner, Report (accuracy per tag) and Diff apply.
type Needle struct {
	Filler   []string  // filler sentences, shuffled and repeated
	Needle   string    // the hidden fact
	Question string    // asked after the document
	Answer   string    // expected substring of a correct answer (graded with evals.Contains)
	Lengths  []int     // haystack sizes in estimated tokens
	Depths   []float64 // relative needle positions: 0 = start, 1 = end
	Seed     uint64
}

// Haystack builds filler text of about tokens tokens (≈ 4 bytes each) and inserts needle
// between sentences at relative position depth.
func Haystack(filler []string, tokens int, needle string, depth float64, rng *rand.Rand) string {
	var sents []string
	size := 0
	for size < tokens*4 {
		perm := rng.Perm(len(filler))
		for _, i := range perm {
			sents = append(sents, filler[i])
			size += len(filler[i]) + 1
			if size >= tokens*4 {
				break
			}
		}
	}
	at := int(depth*float64(len(sents)) + 0.5)
	at = max(0, min(at, len(sents)))
	sents = slices.Insert(sents, at, needle)
	return strings.Join(sents, " ")
}

// Dataset returns one case per (length, depth). Case IDs and tags are "len=N" and
// "depth=D", so Report.ByTag shows accuracy per length and per depth.
func (n Needle) Dataset() *evals.Dataset {
	ds := &evals.Dataset{Name: "needle"}
	for _, l := range n.Lengths {
		for _, d := range n.Depths {
			rng := rand.New(rand.NewPCG(n.Seed, uint64(l)*1000+uint64(d*1000)))
			hay := Haystack(n.Filler, l, n.Needle, d, rng)
			lt, dt := fmt.Sprintf("len=%d", l), fmt.Sprintf("depth=%.2f", d)
			ds.Cases = append(ds.Cases, evals.Case{
				ID:       lt + "/" + dt,
				Input:    "<document>\n" + hay + "\n</document>\n\n" + n.Question,
				Expected: n.Answer,
				Tags:     []string{lt, dt},
			})
		}
	}
	return ds
}

// LLMTarget turns any llm.Client into an evals.Target: one user message in, text out.
func LLMTarget(c llm.Client, model, system string, maxTokens int) evals.Target {
	return func(ctx context.Context, input string) (string, error) {
		resp, err := c.Complete(ctx, llm.Request{Model: model, System: system, MaxTokens: maxTokens,
			Messages: []llm.Message{{Role: llm.RoleUser, Content: input}}})
		if err != nil {
			return "", err
		}
		return resp.Text, nil
	}
}

// NeedleGrid renders a run of Needle.Dataset as a table: rows are depths, columns lengths,
// "+" found, "-" missed, "E" error.
func NeedleGrid(run *evals.Run) string {
	cell := map[[2]string]string{}
	var lens, depths []string
	for _, r := range run.Results {
		l, d, ok := strings.Cut(r.Case.ID, "/")
		if !ok {
			continue
		}
		if !slices.Contains(lens, l) {
			lens = append(lens, l)
		}
		if !slices.Contains(depths, d) {
			depths = append(depths, d)
		}
		mark := "-"
		switch {
		case r.Err != "":
			mark = "E"
		case r.Grade.Pass:
			mark = "+"
		}
		cell[[2]string{d, l}] = mark
	}
	num := func(s string) float64 { _, v, _ := strings.Cut(s, "="); f, _ := strconv.ParseFloat(v, 64); return f }
	slices.SortFunc(lens, func(a, b string) int { return int(num(a) - num(b)) })
	slices.SortFunc(depths, func(a, b string) int {
		switch {
		case num(a) < num(b):
			return -1
		case num(a) > num(b):
			return 1
		}
		return 0
	})
	var b strings.Builder
	fmt.Fprintf(&b, "%-11s", "")
	for _, l := range lens {
		fmt.Fprintf(&b, "%10s", l)
	}
	b.WriteString("\n")
	for _, d := range depths {
		fmt.Fprintf(&b, "%-11s", d)
		for _, l := range lens {
			fmt.Fprintf(&b, "%10s", cell[[2]string{d, l}])
		}
		b.WriteString("\n")
	}
	return b.String()
}
