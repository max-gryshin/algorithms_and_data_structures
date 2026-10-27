package serve

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"
	"time"

	"algorithms_and_data_structures/ai_engineering/gpt"
	"algorithms_and_data_structures/ai_engineering/gpt/generate"
)

// Policy selects how the engine forms batches.
type Policy int

const (
	Continuous Policy = iota // iteration-level scheduling (Orca): admit and retire every step
	Static                   // a batch is admitted only when the previous one has fully finished
)

// EngineConfig sizes the scheduler.
type EngineConfig struct {
	Policy         Policy
	BlockSize      int // tokens per KV block (16 in vLLM)
	NumBlocks      int // KV budget: admission requires the blocks of the whole prompt
	MaxSeqs        int // running sequences at most
	MaxBatchTokens int // tokens per step: decodes (1 each) + prefill chunks
	PrefillChunk   int // prompt tokens per sequence per step (chunked prefill); 0 = whole prompt
	MaxWaiting     int // queue length; Submit fails with ErrOverloaded beyond it (0 = unlimited)
}

func (c EngineConfig) withDefaults() EngineConfig {
	if c.BlockSize <= 0 {
		c.BlockSize = 16
	}
	if c.NumBlocks <= 0 {
		c.NumBlocks = 1024
	}
	if c.MaxSeqs <= 0 {
		c.MaxSeqs = 16
	}
	if c.MaxBatchTokens < c.MaxSeqs {
		c.MaxBatchTokens = max(c.MaxSeqs, 512)
	}
	return c
}

// Request is one generation job in token ids.
type Request struct {
	Prompt    []int
	MaxTokens int
	Sampler   generate.SamplerConfig
	EOS       []int
}

// Event is one generated token, or the final event (Finish != "") with usage.
type Event struct {
	Token            int // −1 in the final event
	Finish           string
	PromptTokens     int
	CompletionTokens int
}

var (
	ErrOverloaded = errors.New("serve: queue is full")
	ErrStopped    = errors.New("serve: engine stopped")
)

type seq struct {
	ctx       context.Context
	req       Request
	tokens    []int // prompt + generated
	computed  int   // tokens whose K/V are in the session
	blocks    []int // block table: KV blocks accounted to this sequence
	maxNew    int
	generated int
	sess      *gpt.Session
	sampler   *generate.Sampler
	out       chan Event
	arrival   time.Time
	last      time.Time
	order     int64
	logits    []float32
}

// Engine is a continuous-batching scheduler. One goroutine (Run) owns every piece of
// state — queue, running set, allocator — so no locks are needed on the hot path;
// Submit talks to it through a channel. Each step it picks a batch: one decode token for
// every running sequence (first, to protect inter-token latency), then prefill chunks
// within the token budget, then admits new requests while KV blocks allow.
//
// Honesty note: the step runs every scheduled sequence through its own gpt.Session
// concurrently (one goroutine per sequence); the matmuls of different sequences are not
// fused into one (a varlen kernel over a paged cache is exercise 🔴). KV blocks are
// accounted by BlockAllocator, the memory itself lives in the sessions.
type Engine struct {
	m       *gpt.Model
	cfg     EngineConfig
	alloc   *BlockAllocator
	in      chan *seq
	done    chan struct{}
	Metrics *Metrics

	mu      sync.Mutex
	waiting int
	stats   EngineStats
}

// EngineStats is a snapshot after the last step.
type EngineStats struct {
	Steps, Preemptions, Running, Waiting, FreeBlocks int
	BatchTokens, BatchSeqs                           int // totals over all steps
}

func NewEngine(m *gpt.Model, cfg EngineConfig) *Engine {
	cfg = cfg.withDefaults()
	return &Engine{m: m, cfg: cfg, alloc: NewBlockAllocator(cfg.NumBlocks, cfg.BlockSize),
		in: make(chan *seq, 64), done: make(chan struct{}), Metrics: NewMetrics()}
}

func (e *Engine) Stats() EngineStats {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.stats
}

// Submit queues req; the returned channel yields one Event per token and a final event,
// then closes. Cancelling ctx retires the sequence at the next step and frees its blocks.
// The channel is buffered for the whole answer, so a slow reader never blocks the engine.
func (e *Engine) Submit(ctx context.Context, req Request) (<-chan Event, error) {
	ctxLen := e.m.Cfg.Context
	if len(req.Prompt) == 0 || len(req.Prompt) >= ctxLen {
		return nil, fmt.Errorf("serve: prompt of %d tokens, context %d", len(req.Prompt), ctxLen)
	}
	maxNew := req.MaxTokens
	if maxNew <= 0 || len(req.Prompt)+maxNew > ctxLen {
		maxNew = ctxLen - len(req.Prompt)
	}
	if BlocksFor(len(req.Prompt)+maxNew, e.cfg.BlockSize) > e.cfg.NumBlocks {
		return nil, fmt.Errorf("serve: request needs more KV blocks than the engine has")
	}
	e.mu.Lock()
	if e.cfg.MaxWaiting > 0 && e.waiting >= e.cfg.MaxWaiting {
		e.mu.Unlock()
		return nil, ErrOverloaded
	}
	e.waiting++
	e.mu.Unlock()
	s := &seq{ctx: ctx, req: req, tokens: slices.Clone(req.Prompt), maxNew: maxNew,
		sampler: generate.NewSampler(req.Sampler), out: make(chan Event, maxNew+1), arrival: time.Now()}
	select {
	case e.in <- s:
		return s.out, nil
	case <-ctx.Done():
		e.mu.Lock()
		e.waiting--
		e.mu.Unlock()
		return nil, ctx.Err()
	case <-e.done:
		return nil, ErrStopped
	}
}

// Run is the scheduler loop; it returns when ctx is cancelled.
func (e *Engine) Run(ctx context.Context) error {
	defer close(e.done)
	var waiting, running []*seq
	var order int64
	finish := func(s *seq, reason string) {
		e.releaseBlocks(s)
		s.out <- Event{Token: -1, Finish: reason, PromptTokens: len(s.req.Prompt), CompletionTokens: s.generated}
		close(s.out)
	}
	defer func() {
		for _, s := range append(waiting, running...) {
			finish(s, "cancelled")
		}
	}()
	for {
		if len(waiting) == 0 && len(running) == 0 { // idle: block, no polling
			select {
			case s := <-e.in:
				waiting = append(waiting, s)
			case <-ctx.Done():
				return ctx.Err()
			}
		}
	drain:
		for {
			select {
			case s := <-e.in:
				waiting = append(waiting, s)
			default:
				break drain
			}
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		// Retire cancelled requests.
		keep := func(list []*seq, queued bool) []*seq {
			return slices.DeleteFunc(list, func(s *seq) bool {
				if s.ctx.Err() == nil {
					return false
				}
				if queued {
					e.addWaiting(-1)
				}
				finish(s, "cancelled")
				return true
			})
		}
		waiting, running = keep(waiting, true), keep(running, false)

		type work struct {
			s *seq
			n int
		}
		var plan []work
		budget := e.cfg.MaxBatchTokens
		// 1. decodes, oldest first; out of blocks → preempt the newest (recompute later).
		for i := 0; i < len(running); i++ {
			s := running[i]
			if len(s.tokens)-s.computed != 1 {
				continue
			}
			for !e.ensureBlocks(s, s.computed+1) {
				v := running[len(running)-1]
				running = running[:len(running)-1]
				e.preempt(v)
				waiting = append([]*seq{v}, waiting...)
				e.addWaiting(1)
				if v == s {
					break
				}
			}
			if i < len(running) && running[i] == s {
				plan = append(plan, work{s, 1})
				budget--
			}
		}
		// 2. prefill chunks of running sequences, then 3. admission.
		chunk := func(s *seq) int {
			n := len(s.tokens) - s.computed
			if e.cfg.Policy == Static {
				return n // classic static batching prefills the whole prompt at once
			}
			if e.cfg.PrefillChunk > 0 {
				n = min(n, e.cfg.PrefillChunk)
			}
			return min(n, budget)
		}
		for _, s := range running {
			if p := len(s.tokens) - s.computed; p > 1 {
				if n := chunk(s); n > 0 && e.ensureBlocks(s, s.computed+n) {
					plan = append(plan, work{s, n})
					budget -= n
				}
			}
		}
		canAdmit := e.cfg.Policy == Continuous || len(running) == 0
		for canAdmit && len(waiting) > 0 && len(running) < e.cfg.MaxSeqs && (budget > 0 || e.cfg.Policy == Static) {
			s := waiting[0]
			if e.alloc.Free() < BlocksFor(len(s.tokens), e.cfg.BlockSize) {
				break // FCFS: the head of the queue waits for memory, nobody overtakes it
			}
			waiting = waiting[1:]
			e.addWaiting(-1)
			order++
			s.order = order
			if s.sess == nil {
				s.sess = e.m.NewSession(1, len(s.req.Prompt)+s.maxNew)
			}
			running = append(running, s)
			if n := chunk(s); n > 0 && e.ensureBlocks(s, s.computed+n) {
				plan = append(plan, work{s, n})
				budget -= n
			}
		}
		if len(plan) == 0 {
			if len(running) == 0 && len(waiting) > 0 {
				return fmt.Errorf("serve: scheduler stuck") // unreachable: Submit checks sizes
			}
			e.mu.Lock()
			e.stats.Running, e.stats.Waiting, e.stats.FreeBlocks = len(running), len(waiting), e.alloc.Free()
			e.mu.Unlock()
			continue
		}
		// Execute the step: every sequence on its own goroutine.
		var wg sync.WaitGroup
		tokens := 0
		for _, w := range plan {
			tokens += w.n
			wg.Go(func() {
				s := w.s
				lg := s.sess.Forward([][]int{s.tokens[s.computed : s.computed+w.n]}).Data()
				s.computed += w.n
				s.logits = lg[len(lg)-e.m.Cfg.Vocab:]
			})
		}
		wg.Wait()
		now := time.Now()
		for _, w := range plan {
			s := w.s
			if s.computed < len(s.tokens) {
				continue // a prefill chunk that did not reach the end of the prompt
			}
			id := s.sampler.Sample(s.logits, s.tokens)
			s.tokens = append(s.tokens, id)
			s.generated++
			if s.generated == 1 {
				e.Metrics.ObserveTTFT(now.Sub(s.arrival))
			} else {
				e.Metrics.ObserveITL(now.Sub(s.last))
			}
			s.last = now
			s.out <- Event{Token: id}
		}
		running = slices.DeleteFunc(running, func(s *seq) bool {
			if s.computed < len(s.tokens)-1 || s.generated == 0 {
				return false
			}
			reason := ""
			switch {
			case slices.Contains(s.req.EOS, s.tokens[len(s.tokens)-1]):
				reason = "stop"
			case s.generated >= s.maxNew:
				reason = "length"
			default:
				return false
			}
			e.Metrics.ObserveRequest(now.Sub(s.arrival), s.generated)
			finish(s, reason)
			return true
		})
		e.mu.Lock()
		e.stats.Steps++
		e.stats.BatchTokens += tokens
		e.stats.BatchSeqs += len(plan)
		e.stats.Running, e.stats.Waiting, e.stats.FreeBlocks = len(running), len(waiting), e.alloc.Free()
		e.mu.Unlock()
	}
}

func (e *Engine) addWaiting(d int) {
	e.mu.Lock()
	e.waiting += d
	e.mu.Unlock()
}

// ensureBlocks grows s's block account to cover n tokens.
func (e *Engine) ensureBlocks(s *seq, n int) bool {
	need := BlocksFor(n, e.cfg.BlockSize) - len(s.blocks)
	if need <= 0 {
		return true
	}
	ids, ok := e.alloc.Alloc(need)
	if !ok {
		return false
	}
	s.blocks = append(s.blocks, ids...)
	return true
}

// releaseBlocks returns the block table of s to the allocator.
func (e *Engine) releaseBlocks(s *seq) {
	e.alloc.Release(s.blocks...)
	s.blocks = nil
}

// preempt frees the KV of s; it will be recomputed from tokens (prompt + generated).
func (e *Engine) preempt(s *seq) {
	e.releaseBlocks(s)
	s.computed = 0
	s.sess.Reset()
	e.mu.Lock()
	e.stats.Preemptions++
	e.mu.Unlock()
}
