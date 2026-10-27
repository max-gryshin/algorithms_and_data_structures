package train

import (
	"bytes"
	"context"
	"math"
	"math/rand/v2"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"algorithms_and_data_structures/ai_engineering/autograd"
	"algorithms_and_data_structures/ai_engineering/gpt"
	"algorithms_and_data_structures/ai_engineering/tensor"
	"algorithms_and_data_structures/ai_engineering/tokenizer"
)

func tinyModel(seed uint64, vocab, ctx int, dropout float32) *gpt.Model {
	cfg := gpt.TinyGPT2(vocab, ctx)
	cfg.Dropout = dropout
	return gpt.New(rand.New(rand.NewPCG(seed, seed)), cfg)
}

func randomData(seed uint64, n, vocab int) *Dataset {
	rng := rand.New(rand.NewPCG(seed, 1))
	ids := make([]int, n)
	for i := range ids {
		ids[i] = rng.IntN(vocab)
	}
	return NewDataset(ids)
}

func grads(m *gpt.Model) []*tensor.Tensor {
	var g []*tensor.Tensor
	for _, p := range m.Parameters() {
		g = append(g, p.Grad)
	}
	return g
}

func sameGrads(t *testing.T, a, b []*tensor.Tensor, tol float64) {
	t.Helper()
	for i := range a {
		if !tensor.AllClose(a[i], b[i], tol, tol) {
			t.Fatalf("gradient %d differs", i)
		}
	}
}

// One batch of 16 rows at once == 4 accumulated micro-batches of 4 rows ==
// 4 data-parallel workers with 4 rows each.
func TestAccumulationAndDataParallelGiveTheSameGradient(t *testing.T) {
	data := randomData(1, 500, 20)
	x, y := data.Batch(rand.New(rand.NewPCG(2, 2)), 16, 12)

	m := tinyModel(3, 20, 12, 0)
	tr := New(m, data, data, Config{Steps: 1, Batch: 16, Seq: 12})
	lossFull := tr.Gradients([][][]int{x}, [][][]int{y})
	full := grads(m)

	zeroGrad(m)
	var xs, ys [][][]int
	for i := 0; i < 16; i += 4 {
		xs, ys = append(xs, x[i:i+4]), append(ys, y[i:i+4])
	}
	lossAcc := tr.Gradients(xs, ys)
	sameGrads(t, full, grads(m), 1e-5)

	zeroGrad(m)
	dp := New(m, data, data, Config{Steps: 1, Batch: 16, Seq: 12, Workers: 4})
	lossDP := dp.Gradients([][][]int{x}, [][][]int{y})
	sameGrads(t, full, grads(m), 1e-5)

	if math.Abs(lossFull-lossAcc) > 1e-5 || math.Abs(lossFull-lossDP) > 1e-5 {
		t.Fatalf("losses %v %v %v", lossFull, lossAcc, lossDP)
	}
	// Loss at init ≈ ln V (Karpathy's first check).
	if math.Abs(lossFull-math.Log(20)) > 0.1 {
		t.Fatalf("initial loss %.3f, want ≈ ln 20 = %.3f", lossFull, math.Log(20))
	}
}

// Train 2k steps straight, or k steps → checkpoint → load → k more: bit-identical
// weights (dropout on), because batches and masks depend only on (seed, step).
func TestResumeIsExact(t *testing.T) {
	data := randomData(4, 400, 16)
	cfg := Config{Steps: 6, Batch: 4, Seq: 8, Seed: 7, Warmup: 2, WeightDecay: 0.1, Clip: 1}
	straight := New(tinyModel(5, 16, 8, 0.1), data, data, cfg)
	for range 6 {
		straight.TrainStep()
	}

	first := New(tinyModel(5, 16, 8, 0.1), data, data, cfg)
	for range 3 {
		first.TrainStep()
	}
	var buf bytes.Buffer
	if err := first.Checkpoint().Save(&buf); err != nil {
		t.Fatal(err)
	}
	c, err := gpt.LoadCheckpoint(&buf)
	if err != nil {
		t.Fatal(err)
	}
	resumed, err := Resume(c, data, data, cfg)
	if err != nil {
		t.Fatal(err)
	}
	for range 3 {
		resumed.TrainStep()
	}
	a, b := straight.Model.Parameters(), resumed.Model.Parameters()
	for i := range a {
		if !slices.Equal(a[i].Data.Data(), b[i].Data.Data()) {
			t.Fatalf("param %d differs after resume", i)
		}
	}
}

// A periodic sequence is learnable to ~0 loss; Run writes the CSV and a checkpoint.
func TestRunLearnsAndLogs(t *testing.T) {
	var ids []int
	for i := range 600 {
		ids = append(ids, i%7)
	}
	data := NewDataset(ids)
	steps := 120
	if raceEnabled {
		steps = 6
	}
	var csv, progress bytes.Buffer
	path := filepath.Join(t.TempDir(), "m.ckpt")
	cfg := Config{Steps: steps, Batch: 8, Seq: 16, MaxLR: 1e-2, Workers: 2, EvalEvery: steps / 2,
		CheckpointPath: path, CSV: &csv, Progress: &progress, Meta: map[string]string{"note": "test"}}
	tr := New(tinyModel(6, 7, 16, 0), data, data, cfg)
	res, err := tr.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if !raceEnabled && (res.ValLoss > 0.05 || res.TrainLoss > 0.1) {
		t.Fatalf("periodic data not learned: train %.3f val %.3f", res.TrainLoss, res.ValLoss)
	}
	lines := strings.Split(strings.TrimSpace(csv.String()), "\n")
	if len(lines) != steps+1 || !strings.HasPrefix(lines[0], "step,lr,train_loss,val_loss") {
		t.Fatalf("csv: %d lines, header %q", len(lines), lines[0])
	}
	c, err := gpt.LoadCheckpointFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if c.Meta["note"] != "test" || c.Meta["step"] == "" {
		t.Fatalf("checkpoint meta %v", c.Meta)
	}
}

func TestEvalWindowsPredictEveryTokenOnce(t *testing.T) {
	for _, n := range []int{2, 5, 17, 64, 100} {
		ids := make([]int, n)
		for i := range ids {
			ids[i] = i % 50
		}
		x, y := NewDataset(ids).EvalWindows(8)
		count := 0
		for i := range y {
			for j, id := range y[i] {
				if id >= 0 {
					count++
					if x[i][j] != (id+49)%50 {
						t.Fatalf("n=%d: target %d after input %d", n, id, x[i][j])
					}
				}
			}
		}
		if count != n-1 {
			t.Fatalf("n=%d: %d predictions, want %d", n, count, n-1)
		}
	}
}

func TestTokenFileAndTokenizerMeta(t *testing.T) {
	path := filepath.Join(t.TempDir(), "train.bin")
	ids := []int{0, 1, 65535, 300}
	if err := WriteTokens(path, ids); err != nil {
		t.Fatal(err)
	}
	got, err := ReadTokens(path)
	if err != nil || len(got) != 4 || got[2] != 65535 || got[3] != 300 {
		t.Fatalf("ReadTokens = %v, %v", got, err)
	}
	if WriteTokens(path, []int{70000}) == nil {
		t.Fatal("id > 65535 accepted")
	}

	text := "кот и пёс, cat & dog"
	bpe, err := tokenizer.Train(text, 270, tokenizer.Options{Pattern: tokenizer.PatternGPT4})
	if err != nil {
		t.Fatal(err)
	}
	for _, tok := range []Tokenizer{NewCharTokenizer(text), bpe} {
		meta, err := TokenizerMeta(tok)
		if err != nil {
			t.Fatal(err)
		}
		back, err := TokenizerFromMeta(meta)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(back.Encode(text), tok.Encode(text)) || back.Decode(tok.Encode(text)) != text {
			t.Fatalf("%T: tokenizer not restored", tok)
		}
	}
}

func zeroGrad(m *gpt.Model) {
	for _, p := range m.Parameters() {
		p.ZeroGrad()
	}
}

// Config.Params trains a subset: the other parameters stay bit-for-bit unchanged,
// and StepOn accepts masked rows (target −1) instead of random windows.
func TestParamsFilterAndStepOn(t *testing.T) {
	data := randomData(5, 300, 12)
	m := tinyModel(6, 12, 8, 0)
	head := func(m *gpt.Model) []*autograd.Var { return []*autograd.Var{m.Tok.W} }
	tr := New(m, data, data, Config{Steps: 3, Batch: 4, Seq: 8, MaxLR: 1e-2, Params: head})
	frozen := append([]float32(nil), m.Blocks[0].Attn.Q.W.Data.Data()...)
	tok := append([]float32(nil), m.Tok.W.Data.Data()...)
	x, y := data.Batch(rand.New(rand.NewPCG(1, 1)), 4, 8)
	for i := range y {
		for j := range 4 {
			y[i][j] = -1 // "prompt": not trained on
		}
	}
	st := tr.StepOn([][][]int{x}, [][][]int{y})
	if st.Tokens != 32 || st.Step != 1 {
		t.Fatalf("stats %+v", st)
	}
	if !slices.Equal(frozen, m.Blocks[0].Attn.Q.W.Data.Data()) {
		t.Fatal("a parameter outside Params changed")
	}
	if slices.Equal(tok, m.Tok.W.Data.Data()) {
		t.Fatal("the selected parameter did not change")
	}
	if c := tr.Checkpoint(); len(c.Extra) != 0 {
		t.Fatal("custom Params must not store unnamed optimizer moments")
	}
}
