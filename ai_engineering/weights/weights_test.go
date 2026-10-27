package weights

import (
	"bytes"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"algorithms_and_data_structures/ai_engineering/gpt"
)

func TestDTypesByHand(t *testing.T) {
	// −6.75 = −1.1011₂·2²: fp32 0xC0D80000, bf16 = upper half 0xC0D8, fp16 1|10001|1011000000.
	if got := math.Float32bits(-6.75); got != 0xC0D80000 {
		t.Fatalf("fp32 %08x", got)
	}
	if F32ToBF16(-6.75) != 0xC0D8 || F32ToF16(-6.75) != 0xC6C0 {
		t.Fatalf("bf16 %04x fp16 %04x", F32ToBF16(-6.75), F32ToF16(-6.75))
	}
	// 0.1 is not representable: bf16 rounds to 0x3DCD = 0.10009765625, fp16 to 0x2E66 = 0.0999755859375.
	if F32ToBF16(0.1) != 0x3DCD || BF16ToF32(0x3DCD) != 0.10009765625 {
		t.Fatalf("bf16(0.1) %04x", F32ToBF16(0.1))
	}
	if F32ToF16(0.1) != 0x2E66 || F16ToF32(0x2E66) != 0.0999755859375 {
		t.Fatalf("fp16(0.1) %04x", F32ToF16(0.1))
	}
	if F32ToF16(70000) != 0x7C00 || F16ToF32(0x0001) != float32(math.Pow(2, -24)) || F32ToF16(65504) != 0x7BFF {
		t.Fatal("fp16 range")
	}
	for h := range 0x7C00 { // every finite positive half survives the round trip
		if F32ToF16(F16ToF32(uint16(h))) != uint16(h) {
			t.Fatalf("round trip %04x", h)
		}
	}
}

func TestSafetensorsRoundTrip(t *testing.T) {
	data := []float32{1, -2.5, 0.1, 3}
	var buf bytes.Buffer
	err := Write(&buf, []Tensor{{"a", F32, []int{2, 2}, data}, {"b", BF16, []int{4}, data}, {"c", F16, []int{4}, data}},
		map[string]string{"format": "pt"})
	if err != nil {
		t.Fatal(err)
	}
	b := buf.Bytes()
	if hl := int(b[0]) | int(b[1])<<8; (hl)%8 != 0 {
		t.Fatalf("header length %d not aligned", hl)
	}
	f, err := Parse(b)
	if err != nil {
		t.Fatal(err)
	}
	if f.Meta["format"] != "pt" || len(f.Names()) != 3 {
		t.Fatalf("meta %v names %v", f.Meta, f.Names())
	}
	for _, n := range []string{"a", "b", "c"} {
		got, _, err := f.Float32(n)
		if err != nil {
			t.Fatal(err)
		}
		for i := range data {
			if math.Abs(float64(got[i]-data[i])) > 1e-3*math.Abs(float64(data[i]))+1e-6 {
				t.Fatalf("%s[%d] = %v, want %v", n, i, got[i], data[i])
			}
		}
	}
	if _, err := Parse(b[:len(b)-1]); err == nil {
		t.Fatal("truncated file accepted")
	}
}

func modelLogits(m *gpt.Model, ids []int) []float32 { return m.Forward([][]int{ids}).Data.Data() }

func maxDiff(a, b []float32) float64 {
	d := 0.0
	for i := range a {
		d = max(d, math.Abs(float64(a[i]-b[i])))
	}
	return d
}

// exportGPT2 writes m with Hugging Face GPT-2 names and layout (fused c_attn).
func exportGPT2(m *gpt.Model) []Tensor {
	p := map[string][]float32{}
	for _, np := range m.NamedParameters() {
		p[np.Name] = np.P.Data.Data()
	}
	d, ff := m.Cfg.DModel, m.Cfg.FFHidden
	ts := []Tensor{{"wte.weight", F32, []int{m.Cfg.Vocab, d}, p["tok.W"]}, {"wpe.weight", F32, []int{m.Cfg.Context, d}, p["pos.W"]},
		{"ln_f.weight", F32, []int{d}, p["normf.g"]}, {"ln_f.bias", F32, []int{d}, p["normf.b"]}}
	for i := range m.Cfg.Layers {
		b, h := fmt.Sprintf("blocks.%d.", i), fmt.Sprintf("h.%d.", i)
		qkv := make([]float32, 0, 3*d*d)
		for r := range d {
			for _, n := range []string{"q", "k", "v"} {
				qkv = append(qkv, p[b+"attn."+n+".W"][r*d:(r+1)*d]...)
			}
		}
		qkvB := append(append(append([]float32{}, p[b+"attn.q.b"]...), p[b+"attn.k.b"]...), p[b+"attn.v.b"]...)
		ts = append(ts, Tensor{h + "attn.c_attn.weight", F32, []int{d, 3 * d}, qkv}, Tensor{h + "attn.c_attn.bias", F32, []int{3 * d}, qkvB},
			Tensor{h + "attn.c_proj.weight", F32, []int{d, d}, p[b+"attn.o.W"]}, Tensor{h + "attn.c_proj.bias", F32, []int{d}, p[b+"attn.o.b"]},
			Tensor{h + "ln_1.weight", F32, []int{d}, p[b+"norm1.g"]}, Tensor{h + "ln_1.bias", F32, []int{d}, p[b+"norm1.b"]},
			Tensor{h + "ln_2.weight", F32, []int{d}, p[b+"norm2.g"]}, Tensor{h + "ln_2.bias", F32, []int{d}, p[b+"norm2.b"]},
			Tensor{h + "mlp.c_fc.weight", F32, []int{d, ff}, p[b+"ffn.up.W"]}, Tensor{h + "mlp.c_fc.bias", F32, []int{ff}, p[b+"ffn.up.b"]},
			Tensor{h + "mlp.c_proj.weight", F32, []int{ff, d}, p[b+"ffn.down.W"]}, Tensor{h + "mlp.c_proj.bias", F32, []int{d}, p[b+"ffn.down.b"]},
			Tensor{h + "attn.bias", F32, []int{1}, []float32{1}}) // HF causal-mask buffer: must be ignored
	}
	return ts
}

// exportLlama writes m the way HF stores Llama: (out, in) matrices, rotate_half q/k rows.
func exportLlama(m *gpt.Model) []Tensor {
	p := map[string][]float32{}
	for _, np := range m.NamedParameters() {
		p[np.Name] = np.P.Data.Data()
	}
	c := m.Cfg
	d, ff, hd := c.DModel, c.FFHidden, c.HeadDim()
	hfLinear := func(w []float32, in, out int, perm []int) []float32 {
		wt := Transpose(w, in, out) // (out, in), our row order
		if perm == nil {
			return wt
		}
		hf := make([]float32, len(wt))
		for r, src := range perm { // our row r = HF row src
			copy(hf[src*in:(src+1)*in], wt[r*in:(r+1)*in])
		}
		return hf
	}
	ts := []Tensor{{"model.embed_tokens.weight", BF16, []int{c.Vocab, d}, p["tok.W"]}, {"model.norm.weight", F32, []int{d}, p["normf.g"]},
		{"lm_head.weight", F32, []int{c.Vocab, d}, Transpose(p["head.W"], d, c.Vocab)}}
	for i := range c.Layers {
		b, h := fmt.Sprintf("blocks.%d.", i), fmt.Sprintf("model.layers.%d.", i)
		ts = append(ts,
			Tensor{h + "input_layernorm.weight", F32, []int{d}, p[b+"norm1.g"]},
			Tensor{h + "post_attention_layernorm.weight", F32, []int{d}, p[b+"norm2.g"]},
			Tensor{h + "self_attn.q_proj.weight", F32, []int{c.Heads * hd, d}, hfLinear(p[b+"attn.q.W"], d, c.Heads*hd, RopePermutation(c.Heads, hd))},
			Tensor{h + "self_attn.k_proj.weight", F32, []int{c.KVHeads * hd, d}, hfLinear(p[b+"attn.k.W"], d, c.KVHeads*hd, RopePermutation(c.KVHeads, hd))},
			Tensor{h + "self_attn.v_proj.weight", F32, []int{c.KVHeads * hd, d}, hfLinear(p[b+"attn.v.W"], d, c.KVHeads*hd, nil)},
			Tensor{h + "self_attn.o_proj.weight", F32, []int{d, c.Heads * hd}, hfLinear(p[b+"attn.o.W"], c.Heads*hd, d, nil)},
			Tensor{h + "mlp.gate_proj.weight", F32, []int{ff, d}, hfLinear(p[b+"ffn.gate.W"], d, ff, nil)},
			Tensor{h + "mlp.up_proj.weight", F32, []int{ff, d}, hfLinear(p[b+"ffn.up.W"], d, ff, nil)},
			Tensor{h + "mlp.down_proj.weight", F32, []int{d, ff}, hfLinear(p[b+"ffn.down.W"], ff, d, nil)})
	}
	return ts
}

func roundTrip(t *testing.T, ts []Tensor) *File {
	var buf bytes.Buffer
	if err := Write(&buf, ts, nil); err != nil {
		t.Fatal(err)
	}
	f, err := Parse(buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	return f
}

func TestLoadGPT2MatchesModel(t *testing.T) {
	src := gpt.New(rand.New(rand.NewPCG(1, 2)), gpt.TinyGPT2(50, 16))
	got, err := LoadGPT2(roundTrip(t, exportGPT2(src)), src.Cfg)
	if err != nil {
		t.Fatal(err)
	}
	ids := []int{3, 1, 4, 1, 5, 9, 2, 6}
	if d := maxDiff(modelLogits(src, ids), modelLogits(got, ids)); d != 0 {
		t.Fatalf("logits differ by %g", d)
	}
}

func TestLoadLlamaMatchesModel(t *testing.T) {
	cfg := gpt.TinyLlama(50, 16)
	src := gpt.New(rand.New(rand.NewPCG(3, 4)), cfg)
	for i := range src.Tok.W.Data.Data() { // bf16 in the file: make the source exactly representable
		src.Tok.W.Data.Data()[i] = BF16ToF32(F32ToBF16(src.Tok.W.Data.Data()[i]))
	}
	got, err := LoadLlama(roundTrip(t, exportLlama(src)), cfg)
	if err != nil {
		t.Fatal(err)
	}
	ids := []int{3, 1, 4, 1, 5, 9, 2, 6}
	if d := maxDiff(modelLogits(src, ids), modelLogits(got, ids)); d > 1e-5 {
		t.Fatalf("logits differ by %g", d)
	}
	ts := exportLlama(src)
	if _, err := LoadLlama(roundTrip(t, ts[:len(ts)-1]), cfg); err == nil {
		t.Fatal("missing tensor accepted")
	}
}

func TestQuantParamsByHand(t *testing.T) {
	// The lecture's example: x = [−1.2, 0.3, 2.54], int8 symmetric: s = 2.54/127 = 0.02.
	p := Symmetric(2.54, 8)
	if math.Abs(float64(p.Scale-0.02)) > 1e-7 || p.Quantize(-1.2) != -60 || p.Quantize(0.3) != 15 || p.Quantize(2.54) != 127 {
		t.Fatalf("symmetric %+v", p)
	}
	// Asymmetric uint8 on [−1.2, 2.54]: s = 3.74/255 ≈ 0.014667, z = round(1.2/s) = 82.
	a := Asymmetric(-1.2, 2.54, 8)
	if a.Zero != 82 || a.Quantize(-1.2) != 0 || a.Quantize(2.54) != 255 || a.Dequantize(a.Quantize(0)) != 0 {
		t.Fatalf("asymmetric %+v q(-1.2)=%d", a, a.Quantize(-1.2))
	}
}

func randMatrix(rng *rand.Rand, out, in int) []float32 {
	w := make([]float32, out*in)
	for i := range w {
		w[i] = float32(rng.NormFloat64()) * 0.02
	}
	return w
}

func relErr(a, b []float32) float64 {
	var num, den float64
	for i := range a {
		num += float64((a[i] - b[i]) * (a[i] - b[i]))
		den += float64(a[i] * a[i])
	}
	return math.Sqrt(num / den)
}

func TestQuantizedMatVec(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	out, in := 96, 128
	w, x := randMatrix(rng, out, in), randMatrix(rng, 1, in)
	ref := make([]float32, out)
	(&F32Matrix{out, in, w}).MulVec(ref, x)
	for _, c := range []struct {
		f      Format
		maxErr float64
		bits   float64
	}{{FormatInt8, 0.01, 8.25}, {FormatInt4, 0.15, 4.5}} {
		m := Quantize(w, out, in, c.f)
		got := make([]float32, out)
		m.MulVec(got, x)
		// the kernel must equal "dequantize, then multiply in float32"
		deq := make([]float32, out)
		(&F32Matrix{out, in, m.Dequantize()}).MulVec(deq, x)
		if d := maxDiff(got, deq); d > 1e-5 {
			t.Fatalf("%s kernel vs dequantized differs by %g", c.f, d)
		}
		e := relErr(ref, got)
		bits := float64(8*m.Bytes()) / float64(out*in)
		t.Logf("%s: relative output error %.4f, %.2f bits/weight", c.f, e, bits)
		if e > c.maxErr || bits > c.bits+0.01 {
			t.Fatalf("%s: error %.4f, %.2f bits/weight", c.f, e, bits)
		}
	}
}

func TestPerChannelBeatsPerTensorWithOutlierRow(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	out, in := 16, 64
	w := randMatrix(rng, out, in)
	for i := range in {
		w[i] *= 100 // one output channel with a 100× larger scale
	}
	errOf := func(m *Int8Matrix) float64 { return relErr(w[in:], m.Dequantize()[in:]) } // the normal rows
	pc, pt := errOf(QuantizeInt8(w, out, in, false)), errOf(QuantizeInt8(w, out, in, true))
	if pc*10 > pt {
		t.Fatalf("per-channel %.4f, per-tensor %.4f", pc, pt)
	}
}

func TestFakeQuantizePerplexity(t *testing.T) {
	cfg := gpt.TinyGPT2(64, 32)
	m := gpt.New(rand.New(rand.NewPCG(9, 10)), cfg)
	ids := make([]int, 64)
	for i := range ids {
		ids[i] = (i * 7) % 64
	}
	base := Perplexity(m, ids)
	q, f := FakeQuantize(m, FormatInt8)
	ppl := Perplexity(m, ids)
	t.Logf("ppl f32 %.4f → int8 %.4f, %d → %d bytes", base, ppl, f, q)
	if math.Abs(ppl-base)/base > 0.01 || q*3 > f {
		t.Fatalf("ppl %.4f vs %.4f, bytes %d vs %d", ppl, base, q, f)
	}
}

func benchMat(b *testing.B, f Format) {
	rng := rand.New(rand.NewPCG(1, 1))
	out, in := 3072, 768 // one GPT-2 MLP matrix
	m := Quantize(randMatrix(rng, out, in), out, in, f)
	x, dst := randMatrix(rng, 1, in), make([]float32, out)
	b.SetBytes(int64(m.Bytes()))
	for b.Loop() {
		m.MulVec(dst, x)
	}
}

func BenchmarkMulVecF32(b *testing.B)  { benchMat(b, FormatF32) }
func BenchmarkMulVecInt8(b *testing.B) { benchMat(b, FormatInt8) }
func BenchmarkMulVecInt4(b *testing.B) { benchMat(b, FormatInt4) }
