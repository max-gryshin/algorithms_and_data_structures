package posttrain

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"algorithms_and_data_structures/ai_engineering/autograd"
	"algorithms_and_data_structures/ai_engineering/gpt"
	"algorithms_and_data_structures/ai_engineering/tensor"
)

func near(t *testing.T, name string, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Fatalf("%s = %.6f, want %.6f", name, got, want)
	}
}

func TestTemplateAndLossMask(t *testing.T) {
	tok := NewChatTokenizer("абвгдежзийклмнопрстуфхцчшщъыьэюя: ?")
	msgs := []Message{{"user", "кот?"}, {"assistant", "да"}}
	if s := Format(msgs, false); s != "<|user|>кот?<|end|><|assistant|>да<|end|>" {
		t.Fatalf("Format = %q", s)
	}
	ex := Tokenize(tok, msgs)
	if !slices.Equal(ex.IDs, tok.Encode(Format(msgs, false))) {
		t.Fatal("Tokenize and Encode(Format) disagree")
	}
	// tokens: <u> к о т ? <e> <a> д а <e>; trained: д а <e>
	x, y := ex.XY(true)
	if len(x) != 9 {
		t.Fatalf("len %d", len(x))
	}
	want := []int{-1, -1, -1, -1, -1, -1, ex.IDs[7], ex.IDs[8], ex.IDs[9]}
	if !slices.Equal(y, want) {
		t.Fatalf("y = %v, want %v", y, want)
	}
	if got := tok.EncodeText("<|end|>"); len(got) == 1 {
		t.Fatal("plain text must not produce a special token")
	}
	if tok.Decode(ex.IDs) != Format(msgs, false) {
		t.Fatal("Decode")
	}
	bx, by := Batch(rand.New(rand.NewPCG(1, 1)), []Example{ex, Tokenize(tok, msgs[:1])}, 4, tok.Special(TokEnd), true)
	for i := range bx {
		if len(bx[i]) != 9 || len(by[i]) != 9 {
			t.Fatal("padding")
		}
	}
}

func TestBradleyTerryByHand(t *testing.T) {
	// r_w = 2, r_l = 0.5: −log σ(1.5) = log(1 + e^−1.5) = 0.2014
	l := BradleyTerryLoss(autograd.Param(tensor.FromSlice([]float32{2}, 1)), autograd.NewVar(tensor.FromSlice([]float32{0.5}, 1)))
	near(t, "BT", float64(l.Item()), 0.201413, 1e-5)
	// Large margins must not overflow.
	big := BradleyTerryLoss(autograd.NewVar(tensor.FromSlice([]float32{-100}, 1)), autograd.NewVar(tensor.FromSlice([]float32{0}, 1)))
	near(t, "BT(−100)", float64(big.Item()), 100, 1e-3)
}

func TestDPOByHandAndGrad(t *testing.T) {
	// log π(y_w) = −10, ref −12 → +2; log π(y_l) = −9, ref −8 → −1; β = 0.1 → z = 0.3
	near(t, "DPO64", DPOLoss64(-10, -9, -12, -8, 0.1), math.Log(1+math.Exp(-0.3)), 1e-12)
	near(t, "DPO64", DPOLoss64(-10, -9, -12, -8, 0.1), 0.554355, 1e-6)
	pw := autograd.Param(tensor.FromSlice([]float32{-10, -3}, 2))
	pl := autograd.Param(tensor.FromSlice([]float32{-9, -4}, 2))
	rw := autograd.NewVar(tensor.FromSlice([]float32{-12, -3}, 2))
	rl := autograd.NewVar(tensor.FromSlice([]float32{-8, -4}, 2))
	l := DPOLoss(pw, pl, rw, rl, 0.1)
	near(t, "DPO", float64(l.Item()), (0.554355+math.Log(2))/2, 1e-5)
	rel, where := autograd.GradCheck(func() *autograd.Var { return DPOLoss(pw, pl, rw, rl, 0.1) }, []*autograd.Var{pw, pl}, 1e-2)
	if rel > 2e-2 {
		t.Fatalf("gradcheck %.3g at %s", rel, where)
	}
}

func TestPPOClipAndGroupAdvantages(t *testing.T) {
	near(t, "clip+", PPOClip(1.5, 1, 0.2), 1.2, 1e-12)
	near(t, "clip−", PPOClip(0.5, -1, 0.2), -0.8, 1e-12)
	near(t, "noclip", PPOClip(0.5, 1, 0.2), 0.5, 1e-12) // pessimistic: worse ratio not clipped
	a := GroupAdvantages([]float64{1, 0, 0, 1, 1, 0, 0, 0})
	near(t, "A+", a[0], 1.290994, 1e-5)
	near(t, "A−", a[1], -0.774597, 1e-5)
	near(t, "KL", KLk3(-1, -1), 0, 1e-12)
}

// LoRA on a whole model: at init the outputs are unchanged, only adapters are trainable.
func TestApplyLoRA(t *testing.T) {
	m := gpt.New(rand.New(rand.NewPCG(1, 1)), gpt.TinyGPT2(10, 8))
	ids := [][]int{{1, 2, 3, 4}}
	before := m.Forward(ids).Data
	ps := ApplyLoRA(m, rand.New(rand.NewPCG(2, 2)), 4, 8)
	if !tensor.AllClose(before, m.Forward(ids).Data, 0, 0) {
		t.Fatal("LoRA changed the model at init")
	}
	// 2 layers × 4 attention matrices 32×32 × r(in+out) = 8 · 4·64 = 2048
	if n := 0; true {
		for _, p := range ps {
			n += p.Data.Size()
		}
		if n != 2048 {
			t.Fatalf("adapter params %d", n)
		}
	}
	MergeLoRA(m)
	if !tensor.AllClose(before, m.Forward(ids).Data, 0, 0) {
		t.Fatal("merge with B = 0 changed the model")
	}
}
