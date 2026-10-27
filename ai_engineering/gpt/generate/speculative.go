package generate

import (
	"context"
	"fmt"
	"math/rand/v2"
	"time"

	"algorithms_and_data_structures/ai_engineering/gpt"
)

// SpecStats describe a speculative generation.
type SpecStats struct {
	Rounds      int // draft-then-verify rounds = target forward passes after the prefill
	Proposed    int // draft tokens proposed
	Accepted    int // draft tokens accepted
	Tokens      int // tokens generated (accepted + one corrected/bonus token per round)
	DraftCalls  int // draft forward passes
	Elapsed     time.Duration
	AcceptRate  float64 // Accepted / Proposed
	TokensRound float64 // Tokens / Rounds
}

// Speculative generates maxTokens tokens after prompt with speculative decoding
// (Leviathan et al. 2023; Chen et al. 2023). Every round:
//
//  1. the cheap draft model proposes k tokens d₁…d_k autoregressively, with
//     distributions q₁…q_k;
//  2. the target model scores the whole proposal in ONE forward pass over k+1 new
//     positions, giving p₁…p_{k+1};
//  3. dᵢ is accepted with probability min(1, pᵢ(dᵢ)/qᵢ(dᵢ)); at the first rejection a
//     token is drawn from the residual norm(max(0, pᵢ − qᵢ)) and the round ends;
//     if all k are accepted, a bonus token is drawn from p_{k+1}.
//
// The output has exactly the distribution of sampling from the target alone (proof
// in L10); with Temperature 0 it is exactly the target's greedy output. The speedup
// comes from step 2: verifying k tokens costs about as much as generating one,
// because decoding is memory-bound.
//
// Both models must share the tokenizer. sc applies to both distributions.
func Speculative(ctx context.Context, target, draft *gpt.Model, prompt []int, k, maxTokens int,
	sc SamplerConfig) ([]int, SpecStats, error) {
	var st SpecStats
	if target.Cfg.Vocab != draft.Cfg.Vocab {
		return nil, st, fmt.Errorf("generate: target vocab %d, draft vocab %d", target.Cfg.Vocab, draft.Cfg.Vocab)
	}
	if len(prompt) == 0 || k < 1 {
		return nil, st, fmt.Errorf("generate: empty prompt or k < 1")
	}
	start := time.Now()
	ctxLen := min(target.Cfg.Context, draft.Cfg.Context)
	ts, ds := target.NewSession(1, ctxLen), draft.NewSession(1, ctxLen)
	sampler := NewSampler(sc)
	rng := rand.New(rand.NewPCG(sc.Seed, 0xacce97))
	seq := append([]int(nil), prompt...)
	v := target.Cfg.Vocab

	// feed sends the tokens a session has not seen yet and returns the logits of the
	// last len(out)/V of them.
	feed := func(s *gpt.Session, toks []int) []float32 {
		return s.Forward([][]int{toks}).Data()
	}
	for len(seq)-len(prompt) < maxTokens {
		if err := ctx.Err(); err != nil {
			return seq[len(prompt):], st, err
		}
		kk := min(k, maxTokens-(len(seq)-len(prompt))-1, ctxLen-len(seq)-1)
		if kk < 0 {
			break // context full
		}
		// 1. Draft: k tokens, keeping q_i for each.
		qs := make([][]float64, kk)
		draftToks := make([]int, kk)
		hist := append([]int(nil), seq...)
		var logits []float32
		if kk > 0 {
			logits = feed(ds, seq[ds.Len():])
			st.DraftCalls++
		}
		for i := range kk {
			qs[i] = sampler.Distribution(logits[len(logits)-v:], hist)
			draftToks[i] = Draw(rng, qs[i])
			hist = append(hist, draftToks[i])
			if i+1 < kk {
				logits = feed(ds, draftToks[i:i+1])
				st.DraftCalls++
			}
		}
		// 2. Target: the unseen tail of seq plus all draft tokens in one pass.
		tl := feed(ts, append(seq[ts.Len():], draftToks...))
		st.Rounds++
		rows := len(tl) / v
		p := func(i int) []float64 { // target distribution for draft position i (0…kk)
			row := tl[(rows-kk-1+i)*v : (rows-kk+i)*v]
			return sampler.Distribution(row, append(seq[:len(seq):len(seq)], draftToks[:i]...))
		}
		// 3. Accept / reject.
		accepted := 0
		var next int
		for ; accepted < kk; accepted++ {
			d := draftToks[accepted]
			st.Proposed++
			tok, ok := Verify(rng, p(accepted), qs[accepted], d)
			if ok {
				st.Accepted++
				continue
			}
			next = tok
			break
		}
		if accepted == kk {
			next = Draw(rng, p(kk)) // bonus token from the target's last row
		}
		seq = append(seq, draftToks[:accepted]...)
		seq = append(seq, next)
		st.Tokens += accepted + 1
		// Roll the caches back to the accepted prefix (the newest token is not fed yet).
		ts.Truncate(min(ts.Len(), len(seq)-1))
		ds.Truncate(min(ds.Len(), len(seq)-1))
	}
	st.Elapsed = time.Since(start)
	if st.Proposed > 0 {
		st.AcceptRate = float64(st.Accepted) / float64(st.Proposed)
	}
	if st.Rounds > 0 {
		st.TokensRound = float64(st.Tokens) / float64(st.Rounds)
	}
	out := seq[len(prompt):]
	if len(out) > maxTokens {
		out = out[:maxTokens]
	}
	return out, st, nil
}

// Verify is the accept/reject rule of speculative sampling for one draft token d
// drawn from q: accept with probability min(1, p(d)/q(d)); otherwise draw from the
// residual distribution norm(max(0, p − q)). It returns the token to emit and whether
// it is the draft token. For d ~ q the returned token is distributed exactly as p.
func Verify(rng *rand.Rand, p, q []float64, d int) (int, bool) {
	if rng.Float64()*q[d] < p[d] { // u < p(d)/q(d), written without division
		return d, true
	}
	res := make([]float64, len(p))
	var sum float64
	for j := range res {
		res[j] = max(0, p[j]-q[j])
		sum += res[j]
	}
	if sum == 0 { // p == q: a rejection has probability 0; guard against rounding
		res = p
	}
	return Draw(rng, res), false
}

// ExpectedTokensPerRound is E[tokens per target pass] = (1 − α^(k+1)) / (1 − α) for an
// acceptance rate α (assumed independent per token) and k draft tokens.
func ExpectedTokensPerRound(alpha float64, k int) float64 {
	if alpha >= 1 {
		return float64(k + 1)
	}
	pow := 1.0
	for range k + 1 {
		pow *= alpha
	}
	return (1 - pow) / (1 - alpha)
}
