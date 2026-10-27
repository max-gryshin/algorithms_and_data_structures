package train

import (
	"context"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"runtime"
	"strconv"
	"sync"
	"time"

	"algorithms_and_data_structures/ai_engineering/autograd"
	"algorithms_and_data_structures/ai_engineering/gpt"
	"algorithms_and_data_structures/ai_engineering/optim"
)

// Config of a training run. Zero values get the defaults of Defaults().
type Config struct {
	Steps int // optimizer steps
	Batch int // sequences per micro-batch (split between Workers)
	Seq   int // tokens per sequence (≤ model context)
	Accum int // micro-batches per optimizer step: effective batch = Batch·Accum sequences

	Workers int // data-parallel goroutines, each with its own model replica
	Threads int // intra-op threads per op (autograd.SetThreads); 0 = GOMAXPROCS/Workers

	MaxLR, MinLR float64 // warmup to MaxLR, cosine decay to MinLR
	Warmup       int     // warmup steps
	WeightDecay  float64 // AdamW λ (matrices only)
	Clip         float64 // global gradient-norm clip; 0 = off

	EvalEvery       int    // evaluate the validation loss every N steps (and at the end); 0 = only at the end
	EvalMaxTokens   int    // cap on validation tokens per evaluation; 0 = whole set
	LogEvery        int    // print a progress line every N steps
	CheckpointPath  string // where to save; "" = no checkpoints
	CheckpointEvery int    // save every N steps (and at the end)
	KeepBest        bool   // save only when the validation loss improves (early stopping)
	Seed            uint64 // batches and dropout masks are functions of (Seed, step)

	// Params selects the trainable parameters (L13): nil trains all of m.Parameters();
	// LoRA passes the adapters only. With a custom set, Workers must be 1 (replicas copy
	// only the base weights) and checkpoints store the weights without optimizer moments.
	Params func(m *gpt.Model) []*autograd.Var

	Meta     map[string]string // copied into every checkpoint (tokenizer, corpus…)
	CSV      io.Writer         // one row per step: step,lr,train_loss,val_loss,grad_norm,update_ratio,tok_per_s,elapsed_s
	Progress io.Writer         // human-readable progress lines
}

// Defaults fills zero fields.
func (c Config) Defaults() Config {
	def := func(p *int, v int) {
		if *p == 0 {
			*p = v
		}
	}
	def(&c.Batch, 16)
	def(&c.Seq, 64)
	def(&c.Accum, 1)
	def(&c.Workers, 1)
	def(&c.Warmup, c.Steps/20)
	def(&c.LogEvery, max(1, c.Steps/20))
	def(&c.CheckpointEvery, c.Steps)
	if c.MaxLR == 0 {
		c.MaxLR = 1e-3
	}
	if c.MinLR == 0 {
		c.MinLR = c.MaxLR / 10
	}
	if c.Threads == 0 {
		c.Threads = max(1, runtime.GOMAXPROCS(0)/c.Workers)
	}
	return c
}

// StepStats describes one optimizer step.
type StepStats struct {
	Step        int // optimizer steps completed (1 after the first step)
	LR          float64
	Loss        float64 // mean training loss over the effective batch
	GradNorm    float64 // global gradient norm before clipping
	UpdateRatio float64 // ‖Δw‖/‖w‖ over all weight matrices (healthy: ~1e-3)
	Tokens      int     // tokens processed in this step
	Duration    time.Duration
}

// Trainer owns the model, the optimizer and the data of one run.
type Trainer struct {
	Model      *gpt.Model
	Opt        *optim.Adam
	Sched      optim.CosineSchedule
	Cfg        Config
	Train, Val *Dataset
	Step       int // optimizer steps done so far

	params   []*autograd.Var
	replicas []*gpt.Model // Workers > 1: one replica per goroutine
	start    time.Time
}

// New prepares a run. The model must have been built with Context ≥ cfg.Seq.
func New(m *gpt.Model, trainSet, valSet *Dataset, cfg Config) *Trainer {
	cfg = cfg.Defaults()
	if cfg.Seq > m.Cfg.Context {
		panic(fmt.Sprintf("train: Seq %d > model context %d", cfg.Seq, m.Cfg.Context))
	}
	if cfg.Batch < cfg.Workers {
		panic(fmt.Sprintf("train: batch %d < %d workers", cfg.Batch, cfg.Workers))
	}
	params := m.Parameters()
	if cfg.Params != nil {
		if cfg.Workers > 1 {
			panic("train: Config.Params needs Workers = 1")
		}
		params = cfg.Params(m)
	}
	t := &Trainer{Model: m, Cfg: cfg, Train: trainSet, Val: valSet, params: params}
	t.Opt = optim.NewAdamW(t.params, cfg.MaxLR, cfg.WeightDecay)
	t.Sched = optim.CosineSchedule{MaxLR: cfg.MaxLR, MinLR: cfg.MinLR, WarmupSteps: cfg.Warmup, TotalSteps: cfg.Steps}
	if cfg.Workers > 1 {
		for range cfg.Workers {
			t.replicas = append(t.replicas, m.Replica())
		}
	}
	return t
}

// TrainStep runs one optimizer step: Accum micro-batches of forward+backward
// (data-parallel over Workers), clip, schedule, AdamW.
func (t *Trainer) TrainStep() StepStats {
	start := time.Now()
	var xs, ys [][][]int
	for micro := range t.Cfg.Accum {
		x, y := t.Train.Batch(BatchRNG(t.Cfg.Seed, t.Step, micro), t.Cfg.Batch, t.Cfg.Seq)
		xs, ys = append(xs, x), append(ys, y)
	}
	st := t.StepOn(xs, ys)
	st.Duration = time.Since(start)
	return st
}

// StepOn is one optimizer step on the given micro-batches (rows of inputs and
// targets, target −1 = not trained on) instead of random windows of t.Train — the
// entry point for data that is not a token stream, e.g. SFT examples with a loss mask (L13).
func (t *Trainer) StepOn(xs, ys [][][]int) StepStats {
	start := time.Now()
	old := autograd.SetThreads(t.Cfg.Threads)
	defer autograd.SetThreads(old)

	t.Opt.ZeroGrad()
	loss := t.Gradients(xs, ys)
	tokens := 0
	for _, x := range xs {
		for _, row := range x {
			tokens += len(row)
		}
	}

	st := StepStats{Loss: loss, Tokens: tokens}
	if t.Cfg.Clip > 0 {
		st.GradNorm = optim.ClipGradNorm(t.params, t.Cfg.Clip)
	} else {
		st.GradNorm = optim.GradNorm(t.params)
	}
	st.LR = t.Sched.LR(t.Step)
	t.Opt.SetLR(st.LR)
	before := t.snapshot()
	t.Opt.Step()
	st.UpdateRatio = t.updateRatio(before)
	t.Step++
	st.Step = t.Step // steps completed, counting this one
	st.Duration = time.Since(start)
	return st
}

// Gradients accumulates into the model's .Grad the gradient of the mean loss over
// all micro-batches (each micro-batch is a list of rows). It returns that mean loss.
//
// Gradient accumulation: the loss of each micro-batch is divided by their number, so
// the accumulated sum equals the gradient of the average — as if the whole effective
// batch had gone through at once.
//
// Data parallelism: the rows of a micro-batch are split between Workers replicas;
// worker w's loss is weighted by rows_w / rows, and the all-reduce sums the replica
// gradients — together that is the average gradient over the whole batch.
func (t *Trainer) Gradients(xs, ys [][][]int) float64 {
	accum := float32(len(xs))
	var total float64
	if len(t.replicas) == 0 {
		for micro := range xs {
			t.Model.Train(dropoutRNG(t.Cfg.Seed, t.Step, micro, 0))
			loss := t.Model.Loss(xs[micro], ys[micro])
			total += float64(loss.Item())
			autograd.Scale(loss, 1/accum).Backward()
		}
		t.Model.Eval()
		return total / float64(accum)
	}

	w := len(t.replicas)
	losses := make([]float64, w)
	for micro := range xs {
		rows := len(xs[micro])
		var wg sync.WaitGroup
		for i, r := range t.replicas {
			lo, hi := i*rows/w, (i+1)*rows/w
			wg.Go(func() {
				r.Train(dropoutRNG(t.Cfg.Seed, t.Step, micro, i+1))
				loss := r.Loss(xs[micro][lo:hi], ys[micro][lo:hi])
				share := float32(hi-lo) / float32(rows)
				losses[i] += float64(loss.Item() * share)
				autograd.Scale(loss, share/accum).Backward()
				r.Eval()
			})
		}
		wg.Wait()
	}
	t.allReduce()
	for _, l := range losses {
		total += l
	}
	return total / float64(accum)
}

// allReduce sums the replica gradients into the master model and clears them.
// Parameters are split between goroutines; each sums its parameters over the
// replicas in a fixed order, so the result is deterministic.
func (t *Trainer) allReduce() {
	reps := make([][]*autograd.Var, len(t.replicas))
	for i, r := range t.replicas {
		reps[i] = r.Parameters()
	}
	var wg sync.WaitGroup
	for k, p := range t.params {
		wg.Go(func() {
			for _, rp := range reps {
				g := rp[k].Grad
				if g == nil {
					continue
				}
				if p.Grad == nil {
					p.Grad = g
				} else {
					dst, src := p.Grad.Data(), g.Data()
					for i := range dst {
						dst[i] += src[i]
					}
				}
				rp[k].Grad = nil
			}
		})
	}
	wg.Wait()
}

func dropoutRNG(seed uint64, step, micro, worker int) *rand.Rand {
	return rand.New(rand.NewPCG(seed^0x9e3779b97f4a7c15, uint64(step)<<20|uint64(micro)<<8|uint64(worker)))
}

// snapshot copies the weight matrices (for the update/data ratio).
func (t *Trainer) snapshot() [][]float32 {
	s := make([][]float32, len(t.params))
	for k, p := range t.params {
		if p.Data.Dims() >= 2 {
			s[k] = append([]float32(nil), p.Data.Data()...)
		}
	}
	return s
}

// updateRatio is ‖Δw‖/‖w‖ over all weight matrices — how big a step is relative to
// the weights. Karpathy's rule of thumb: about 1e-3; 1e-1 means the lr is far too
// high, 1e-5 that the model barely learns.
func (t *Trainer) updateRatio(before [][]float32) float64 {
	var dw, w float64
	for k, p := range t.params {
		if before[k] == nil {
			continue
		}
		for i, x := range p.Data.Data() {
			d := float64(x - before[k][i])
			dw += d * d
			w += float64(before[k][i]) * float64(before[k][i])
		}
	}
	if w == 0 {
		return 0
	}
	return math.Sqrt(dw / w)
}

// EvalLoss is the mean next-token loss (nats per token) on the validation set,
// computed with EvalWindows in eval mode (no dropout).
func (t *Trainer) EvalLoss() float64 {
	return EvalLoss(t.Model, t.Val, t.Cfg.Seq, t.Cfg.Batch, t.Cfg.EvalMaxTokens)
}

// EvalLoss evaluates m on d with windows of seq tokens, batch windows at a time.
// maxTokens > 0 evaluates only the first maxTokens tokens of d.
func EvalLoss(m *gpt.Model, d *Dataset, seq, batch, maxTokens int) float64 {
	if maxTokens > 0 && len(d.Tokens) > maxTokens {
		d = &Dataset{Tokens: d.Tokens[:maxTokens]}
	}
	x, y := d.EvalWindows(seq)
	var nll float64
	var count int
	for i := 0; i < len(x); i += batch {
		j := min(i+batch, len(x))
		n := 0
		for _, row := range y[i:j] {
			for _, id := range row {
				if id >= 0 {
					n++
				}
			}
		}
		nll += float64(m.Loss(x[i:j], y[i:j]).Item()) * float64(n)
		count += n
	}
	return nll / float64(count)
}

// Result summarizes Run.
type Result struct {
	Steps        int
	TrainLoss    float64 // last step
	ValLoss      float64 // last evaluation
	BestValLoss  float64
	BestStep     int
	Elapsed      time.Duration
	TokensPerSec float64
}

// Run trains until Cfg.Steps (continuing from t.Step), evaluating, logging and
// checkpointing as configured. On ctx cancellation it saves a checkpoint and returns.
func (t *Trainer) Run(ctx context.Context) (Result, error) {
	t.start = time.Now()
	res := Result{BestValLoss: math.Inf(1)}
	if t.Cfg.CSV != nil && t.Step == 0 {
		fmt.Fprintln(t.Cfg.CSV, "step,lr,train_loss,val_loss,grad_norm,update_ratio,tok_per_s,elapsed_s")
	}
	tokens := 0
	for t.Step < t.Cfg.Steps {
		if err := ctx.Err(); err != nil {
			if t.Cfg.KeepBest {
				return res, err // the best checkpoint is already on disk
			}
			return res, errorsJoin(err, t.checkpoint())
		}
		st := t.TrainStep()
		tokens += st.Tokens
		res.TrainLoss, res.Steps = st.Loss, t.Step
		val := math.NaN()
		last := t.Step == t.Cfg.Steps
		if last || (t.Cfg.EvalEvery > 0 && t.Step%t.Cfg.EvalEvery == 0) {
			val = t.EvalLoss()
			res.ValLoss = val
			if val < res.BestValLoss {
				res.BestValLoss, res.BestStep = val, t.Step
			}
		}
		elapsed := time.Since(t.start)
		tps := float64(st.Tokens) / st.Duration.Seconds()
		if t.Cfg.CSV != nil {
			vs := ""
			if !math.IsNaN(val) {
				vs = strconv.FormatFloat(val, 'f', 4, 64)
			}
			fmt.Fprintf(t.Cfg.CSV, "%d,%.3e,%.4f,%s,%.4f,%.3e,%.0f,%.2f\n",
				st.Step, st.LR, st.Loss, vs, st.GradNorm, st.UpdateRatio, tps, elapsed.Seconds())
		}
		if t.Cfg.Progress != nil && (st.Step%t.Cfg.LogEvery == 0 || last || !math.IsNaN(val)) {
			line := fmt.Sprintf("step %5d  lr %.2e  loss %.4f  |g| %.3f  upd/w %.1e  %6.0f tok/s",
				st.Step, st.LR, st.Loss, st.GradNorm, st.UpdateRatio, tps)
			if !math.IsNaN(val) {
				line += fmt.Sprintf("  val %.4f", val)
			}
			fmt.Fprintln(t.Cfg.Progress, line)
		}
		save := last || t.Step%t.Cfg.CheckpointEvery == 0
		if t.Cfg.KeepBest {
			save = !math.IsNaN(val) && val <= res.BestValLoss
		}
		if t.Cfg.CheckpointPath != "" && save {
			if err := t.checkpoint(); err != nil {
				return res, err
			}
		}
	}
	res.Elapsed = time.Since(t.start)
	res.TokensPerSec = float64(tokens) / res.Elapsed.Seconds()
	return res, nil
}

func errorsJoin(a, b error) error {
	if b == nil {
		return a
	}
	return fmt.Errorf("%w (checkpoint: %v)", a, b)
}

func (t *Trainer) checkpoint() error {
	if t.Cfg.CheckpointPath == "" {
		return nil
	}
	return t.Checkpoint().SaveFile(t.Cfg.CheckpointPath)
}

// Checkpoint packs the model, the optimizer moments and the step counter.
func (t *Trainer) Checkpoint() *gpt.Checkpoint {
	meta := map[string]string{"step": strconv.Itoa(t.Step), "seed": strconv.FormatUint(t.Cfg.Seed, 10)}
	for k, v := range t.Cfg.Meta {
		meta[k] = v
	}
	n, m, v := t.Opt.State()
	extra := map[string][]float64{"adam.t": {float64(n)}}
	if t.Cfg.Params != nil {
		return &gpt.Checkpoint{Model: t.Model, Meta: meta} // moments of a custom set have no names
	}
	for k, np := range t.Model.NamedParameters() {
		extra["adam.m."+np.Name] = m[k]
		extra["adam.v."+np.Name] = v[k]
	}
	return &gpt.Checkpoint{Model: t.Model, Meta: meta, Extra: extra}
}

// Resume rebuilds a trainer from a checkpoint written by Run: the same weights,
// optimizer moments and step, so training continues exactly where it stopped.
func Resume(c *gpt.Checkpoint, trainSet, valSet *Dataset, cfg Config) (*Trainer, error) {
	t := New(c.Model, trainSet, valSet, cfg)
	step, err := strconv.Atoi(c.Meta["step"])
	if err != nil {
		return nil, fmt.Errorf("train: checkpoint step: %w", err)
	}
	named := c.Model.NamedParameters()
	m, v := make([][]float64, len(named)), make([][]float64, len(named))
	for k, np := range named {
		m[k], v[k] = c.Extra["adam.m."+np.Name], c.Extra["adam.v."+np.Name]
	}
	if len(c.Extra["adam.t"]) != 1 {
		return nil, fmt.Errorf("train: checkpoint has no optimizer state")
	}
	if err := t.Opt.SetState(int(c.Extra["adam.t"][0]), m, v); err != nil {
		return nil, err
	}
	t.Step = step
	return t, nil
}
