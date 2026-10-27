package gpt

import (
	"fmt"
	"math/rand/v2"

	"algorithms_and_data_structures/ai_engineering/autograd"
	"algorithms_and_data_structures/ai_engineering/nn"
)

// NamedParam is a parameter with its path in the model, e.g. "blocks.0.attn.q.W".
// Names are the keys of a checkpoint (L9) and the hook for loading real weights (L12):
// a loader maps "transformer.h.0.attn.c_attn.weight" (GPT-2) or
// "model.layers.0.self_attn.q_proj.weight" (Llama) onto these names, transposing
// (out, in) matrices into our (in, out) layout.
type NamedParam struct {
	Name string
	P    *autograd.Var
}

// NamedParameters lists every parameter in the order of Parameters().
func (m *Model) NamedParameters() []NamedParam {
	var out []NamedParam
	add := func(name string, p *autograd.Var) {
		if p != nil {
			out = append(out, NamedParam{name, p})
		}
	}
	linear := func(prefix string, l *nn.Linear) {
		add(prefix+".W", l.W)
		add(prefix+".b", l.B)
	}
	norm := func(prefix string, l nn.Layer) {
		switch n := l.(type) {
		case *nn.LayerNorm:
			add(prefix+".g", n.G)
			add(prefix+".b", n.B)
		case *nn.RMSNorm:
			add(prefix+".g", n.G)
		default:
			panic(fmt.Sprintf("gpt: unknown norm %T", l))
		}
	}
	add("tok.W", m.Tok.W)
	if m.Pos != nil {
		add("pos.W", m.Pos.W)
	}
	for i, b := range m.Blocks {
		p := fmt.Sprintf("blocks.%d.", i)
		norm(p+"norm1", b.Norm1)
		linear(p+"attn.q", b.Attn.Q)
		linear(p+"attn.k", b.Attn.K)
		linear(p+"attn.v", b.Attn.V)
		linear(p+"attn.o", b.Attn.O)
		norm(p+"norm2", b.Norm2)
		switch f := b.FFN.(type) {
		case *nn.FFN:
			linear(p+"ffn.up", f.Up)
			linear(p+"ffn.down", f.Down)
		case *nn.SwiGLU:
			linear(p+"ffn.gate", f.Gate)
			linear(p+"ffn.up", f.Up)
			linear(p+"ffn.down", f.Down)
		default:
			panic(fmt.Sprintf("gpt: unknown FFN %T", b.FFN))
		}
	}
	norm("normf", m.NormF)
	if m.Head != nil {
		add("head.W", m.Head.W)
	}
	return out
}

// Replica returns a model that shares every weight tensor with m but has its own
// gradients. Data-parallel training (L9) runs one replica per goroutine: forward and
// backward only read the shared weights, each replica accumulates into its own
// .Grad, and the optimizer updates the shared weights after the all-reduce.
func (m *Model) Replica() *Model {
	r := New(rand.New(rand.NewPCG(0, 0)), m.Cfg) // the random init is replaced below
	src, dst := m.Parameters(), r.Parameters()
	for i := range src {
		dst[i].Data = src[i].Data
	}
	return r
}
