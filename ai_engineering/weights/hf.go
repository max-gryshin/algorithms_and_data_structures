package weights

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"algorithms_and_data_structures/ai_engineering/gpt"
	"algorithms_and_data_structures/ai_engineering/nn"
)

// GPT2Config reads a Hugging Face GPT-2 config.json.
func GPT2Config(js []byte) (gpt.Config, error) {
	var c struct {
		Vocab   int     `json:"vocab_size"`
		Ctx     int     `json:"n_positions"`
		D       int     `json:"n_embd"`
		L       int     `json:"n_layer"`
		H       int     `json:"n_head"`
		Inner   *int    `json:"n_inner"`
		Eps     float32 `json:"layer_norm_epsilon"`
		ActFunc string  `json:"activation_function"`
	}
	if err := json.Unmarshal(js, &c); err != nil {
		return gpt.Config{}, err
	}
	cfg := gpt.GPT2Small
	cfg.Name, cfg.Vocab, cfg.Context, cfg.DModel, cfg.Layers, cfg.Heads, cfg.KVHeads = "gpt2", c.Vocab, c.Ctx, c.D, c.L, c.H, c.H
	cfg.FFHidden, cfg.NormEps = 4*c.D, c.Eps
	if c.Inner != nil {
		cfg.FFHidden = *c.Inner
	}
	if c.ActFunc != "" && c.ActFunc != "gelu_new" {
		return cfg, fmt.Errorf("weights: activation %q, our GELU is gelu_new (tanh)", c.ActFunc)
	}
	return cfg, cfg.Validate()
}

// LlamaConfig reads a Hugging Face Llama-family config.json (Llama 2, TinyLlama, SmolLM).
func LlamaConfig(js []byte) (gpt.Config, error) {
	var c struct {
		Vocab   int             `json:"vocab_size"`
		D       int             `json:"hidden_size"`
		FF      int             `json:"intermediate_size"`
		L       int             `json:"num_hidden_layers"`
		H       int             `json:"num_attention_heads"`
		KV      int             `json:"num_key_value_heads"`
		Ctx     int             `json:"max_position_embeddings"`
		Eps     float32         `json:"rms_norm_eps"`
		Theta   float64         `json:"rope_theta"`
		Tied    bool            `json:"tie_word_embeddings"`
		Scaling json.RawMessage `json:"rope_scaling"`
	}
	if err := json.Unmarshal(js, &c); err != nil {
		return gpt.Config{}, err
	}
	if len(c.Scaling) > 0 && string(c.Scaling) != "null" {
		return gpt.Config{}, fmt.Errorf("weights: rope_scaling %s is not supported", c.Scaling)
	}
	cfg := gpt.Config{Name: "llama", Vocab: c.Vocab, Context: c.Ctx, DModel: c.D, Layers: c.L, Heads: c.H,
		KVHeads: c.KV, FFHidden: c.FF, Pos: gpt.PosRoPE, Norm: nn.NormRMS, FFN: nn.FFNSwiGLU, Tied: c.Tied,
		NormEps: c.Eps, RoPETheta: c.Theta}
	return cfg, cfg.Validate()
}

// loader fills the parameters of a model by our names.
type loader struct {
	f      *File
	params map[string]gpt.NamedParam
	used   map[string]bool
}

func newLoader(f *File, m *gpt.Model) *loader {
	l := &loader{f: f, params: map[string]gpt.NamedParam{}, used: map[string]bool{}}
	for _, np := range m.NamedParameters() {
		l.params[np.Name] = np
	}
	return l
}

// read returns tensor `hf` and checks its shape.
func (l *loader) read(hf string, shape ...int) ([]float32, error) {
	d, s, err := l.f.Float32(hf)
	if err != nil {
		return nil, err
	}
	if !slices.Equal(s, shape) {
		return nil, fmt.Errorf("weights: %s has shape %v, want %v", hf, s, shape)
	}
	l.used[hf] = true
	return d, nil
}

// set copies data into our parameter `ours`.
func (l *loader) set(ours string, data []float32) error {
	np, ok := l.params[ours]
	if !ok {
		return fmt.Errorf("weights: model has no parameter %s", ours)
	}
	dst := np.P.Data.Data()
	if len(dst) != len(data) {
		return fmt.Errorf("weights: %s: %d values for %v", ours, len(data), np.P.Data.Shape())
	}
	copy(dst, data)
	delete(l.params, ours)
	return nil
}

func (l *loader) copy(hf, ours string, shape ...int) error {
	d, err := l.read(hf, shape...)
	if err != nil {
		return err
	}
	return l.set(ours, d)
}

// finish reports parameters left unset and tensors of the file nobody read.
func (l *loader) finish(ignore func(string) bool) error {
	for name := range l.params {
		return fmt.Errorf("weights: parameter %s not found in the file", name)
	}
	for _, name := range l.f.Names() {
		if !l.used[name] && !ignore(name) {
			return fmt.Errorf("weights: unexpected tensor %s (wrong mapping?)", name)
		}
	}
	return nil
}

// Transpose turns a row-major (rows, cols) matrix into (cols, rows).
func Transpose(w []float32, rows, cols int) []float32 {
	out := make([]float32, len(w))
	for r := range rows {
		for c := range cols {
			out[c*rows+r] = w[r*cols+c]
		}
	}
	return out
}

// LoadGPT2 fills a model with the tensors of a Hugging Face GPT-2 checkpoint:
//
//	wte.weight (V, d)              → tok.W          wpe.weight (n_ctx, d) → pos.W
//	h.i.ln_1.{weight,bias}         → blocks.i.norm1.{g,b}
//	h.i.attn.c_attn.weight (d, 3d) → split columns into attn.q.W | attn.k.W | attn.v.W
//	h.i.attn.c_proj.weight (d, d)  → attn.o.W        (Conv1D is already (in, out))
//	h.i.mlp.c_fc / c_proj          → ffn.up / ffn.down
//	ln_f                           → normf
//
// The causal-mask buffers h.i.attn.bias and a tied lm_head.weight are ignored.
func LoadGPT2(f *File, cfg gpt.Config) (*gpt.Model, error) {
	m := gpt.NewEmpty(cfg)
	l := newLoader(f, m)
	pre := ""
	if _, ok := f.Tensors["transformer.wte.weight"]; ok {
		pre = "transformer."
	}
	d, ff := cfg.DModel, cfg.FFHidden
	if err := l.copy(pre+"wte.weight", "tok.W", cfg.Vocab, d); err != nil {
		return nil, err
	}
	if err := l.copy(pre+"wpe.weight", "pos.W", cfg.Context, d); err != nil {
		return nil, err
	}
	for i := range cfg.Layers {
		h, b := fmt.Sprintf("%sh.%d.", pre, i), fmt.Sprintf("blocks.%d.", i)
		qkv, err := l.read(h+"attn.c_attn.weight", d, 3*d)
		if err != nil {
			return nil, err
		}
		qkvB, err := l.read(h+"attn.c_attn.bias", 3*d)
		if err != nil {
			return nil, err
		}
		for j, name := range []string{"q", "k", "v"} {
			w := make([]float32, d*d)
			for r := range d {
				copy(w[r*d:(r+1)*d], qkv[r*3*d+j*d:r*3*d+(j+1)*d])
			}
			if err := l.set(b+"attn."+name+".W", w); err != nil {
				return nil, err
			}
			if err := l.set(b+"attn."+name+".b", qkvB[j*d:(j+1)*d]); err != nil {
				return nil, err
			}
		}
		for _, c := range []struct {
			hf, ours string
			shape    []int
		}{
			{"ln_1.weight", "norm1.g", []int{d}}, {"ln_1.bias", "norm1.b", []int{d}},
			{"ln_2.weight", "norm2.g", []int{d}}, {"ln_2.bias", "norm2.b", []int{d}},
			{"attn.c_proj.weight", "attn.o.W", []int{d, d}}, {"attn.c_proj.bias", "attn.o.b", []int{d}},
			{"mlp.c_fc.weight", "ffn.up.W", []int{d, ff}}, {"mlp.c_fc.bias", "ffn.up.b", []int{ff}},
			{"mlp.c_proj.weight", "ffn.down.W", []int{ff, d}}, {"mlp.c_proj.bias", "ffn.down.b", []int{d}},
		} {
			if err := l.copy(h+c.hf, b+c.ours, c.shape...); err != nil {
				return nil, err
			}
		}
	}
	if err := l.copy(pre+"ln_f.weight", "normf.g", d); err != nil {
		return nil, err
	}
	if err := l.copy(pre+"ln_f.bias", "normf.b", d); err != nil {
		return nil, err
	}
	return m, l.finish(func(n string) bool {
		return strings.HasSuffix(n, ".attn.bias") || strings.HasSuffix(n, ".attn.masked_bias") || n == "lm_head.weight"
	})
}

// RopePermutation maps Hugging Face's rotate_half layout to our adjacent pairs. HF
// rotates pairs (j, j + d_head/2) of every head; we rotate (2i, 2i+1) with the same
// frequency θ^(−2i/d_head). So our output column h·d_head + 2i is HF row h·d_head + i
// and column 2i+1 is HF row h·d_head + i + d_head/2. q·k does not change when q and
// k are permuted the same way, so attention is exactly the same.
func RopePermutation(heads, headDim int) []int {
	p := make([]int, heads*headDim)
	for h := range heads {
		for i := range headDim / 2 {
			p[h*headDim+2*i] = h*headDim + i
			p[h*headDim+2*i+1] = h*headDim + i + headDim/2
		}
	}
	return p
}

// LoadLlama fills a model with a Hugging Face Llama checkpoint. HF nn.Linear stores
// (out, in); our Linear is (in, out) → every matrix is transposed; q and k get the
// RoPE permutation.
func LoadLlama(f *File, cfg gpt.Config) (*gpt.Model, error) {
	m := gpt.NewEmpty(cfg)
	l := newLoader(f, m)
	d, ff, hd := cfg.DModel, cfg.FFHidden, cfg.HeadDim()
	kvh := cfg.KVHeads
	if kvh == 0 {
		kvh = cfg.Heads
	}
	// linear reads HF (out, in), optionally permutes rows, stores (in, out).
	linear := func(hf, ours string, out, in int, perm []int) error {
		w, err := l.read(hf, out, in)
		if err != nil {
			return err
		}
		if perm != nil {
			pw := make([]float32, len(w))
			for r, src := range perm {
				copy(pw[r*in:(r+1)*in], w[src*in:(src+1)*in])
			}
			w = pw
		}
		return l.set(ours, Transpose(w, out, in))
	}
	if err := l.copy("model.embed_tokens.weight", "tok.W", cfg.Vocab, d); err != nil {
		return nil, err
	}
	pq, pk := RopePermutation(cfg.Heads, hd), RopePermutation(kvh, hd)
	for i := range cfg.Layers {
		h, b := fmt.Sprintf("model.layers.%d.", i), fmt.Sprintf("blocks.%d.", i)
		steps := []error{
			l.copy(h+"input_layernorm.weight", b+"norm1.g", d),
			l.copy(h+"post_attention_layernorm.weight", b+"norm2.g", d),
			linear(h+"self_attn.q_proj.weight", b+"attn.q.W", cfg.Heads*hd, d, pq),
			linear(h+"self_attn.k_proj.weight", b+"attn.k.W", kvh*hd, d, pk),
			linear(h+"self_attn.v_proj.weight", b+"attn.v.W", kvh*hd, d, nil),
			linear(h+"self_attn.o_proj.weight", b+"attn.o.W", d, cfg.Heads*hd, nil),
			linear(h+"mlp.gate_proj.weight", b+"ffn.gate.W", ff, d, nil),
			linear(h+"mlp.up_proj.weight", b+"ffn.up.W", ff, d, nil),
			linear(h+"mlp.down_proj.weight", b+"ffn.down.W", d, ff, nil),
		}
		for _, err := range steps {
			if err != nil {
				return nil, err
			}
		}
	}
	if err := l.copy("model.norm.weight", "normf.g", d); err != nil {
		return nil, err
	}
	if !cfg.Tied {
		if err := linear("lm_head.weight", "head.W", cfg.Vocab, d, nil); err != nil {
			return nil, err
		}
	}
	return m, l.finish(func(n string) bool {
		return strings.HasSuffix(n, "rotary_emb.inv_freq") || (cfg.Tied && n == "lm_head.weight")
	})
}

// LoadGPT2Dir loads dir/config.json (optional: GPT-2 124M by default) and dir/model.safetensors.
func LoadGPT2Dir(dir string) (*gpt.Model, error) {
	cfg := gpt.GPT2Small
	if js, err := os.ReadFile(filepath.Join(dir, "config.json")); err == nil {
		if cfg, err = GPT2Config(js); err != nil {
			return nil, err
		}
	}
	f, err := Open(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return LoadGPT2(f, cfg)
}
