// Package autograd implements reverse-mode automatic differentiation.
//
// Two levels of granularity live side by side:
//   - Scalar (scalar.go): one node per number, micrograd-style. Used in L3 to see
//     backprop "by hand" on tiny expressions.
//   - Var (var.go, ops.go): one node per tensor. Used from L4 on to build real
//     networks; every later lecture (attention, GPT, LoRA, ...) is built on it.
package autograd

import (
	"fmt"
	"math"
)

// Scalar is a node of a computational graph over single float64 numbers.
// Data is the value computed on the forward pass, Grad is dL/dData filled by Backward.
type Scalar struct {
	Data  float64
	Grad  float64
	Label string

	prev     []*Scalar
	op       string
	backward func() // propagates s.Grad into s.prev[*].Grad
}

// S creates a leaf node (an input or a parameter).
func S(x float64) *Scalar { return &Scalar{Data: x} }

// L creates a labelled leaf node; labels only help when printing graphs.
func L(label string, x float64) *Scalar { return &Scalar{Data: x, Label: label} }

func (s *Scalar) String() string {
	return fmt.Sprintf("Scalar(%s data=%.4f grad=%.4f)", s.Label, s.Data, s.Grad)
}

func newScalar(data float64, op string, prev ...*Scalar) *Scalar {
	return &Scalar{Data: data, op: op, prev: prev}
}

// Add: d(a+b)/da = 1, d(a+b)/db = 1 — the gradient is copied to both inputs.
func (a *Scalar) Add(b *Scalar) *Scalar {
	out := newScalar(a.Data+b.Data, "+", a, b)
	out.backward = func() {
		a.Grad += out.Grad
		b.Grad += out.Grad
	}
	return out
}

// Mul: d(a·b)/da = b, d(a·b)/db = a — each input gets the other one times the upstream gradient.
func (a *Scalar) Mul(b *Scalar) *Scalar {
	out := newScalar(a.Data*b.Data, "*", a, b)
	out.backward = func() {
		a.Grad += b.Data * out.Grad
		b.Grad += a.Data * out.Grad
	}
	return out
}

// Pow raises to a constant power: d(aⁿ)/da = n·aⁿ⁻¹.
func (a *Scalar) Pow(n float64) *Scalar {
	out := newScalar(math.Pow(a.Data, n), fmt.Sprintf("^%g", n), a)
	out.backward = func() {
		a.Grad += n * math.Pow(a.Data, n-1) * out.Grad
	}
	return out
}

func (a *Scalar) Neg() *Scalar          { return a.Mul(S(-1)) }
func (a *Scalar) Sub(b *Scalar) *Scalar { return a.Add(b.Neg()) }
func (a *Scalar) Div(b *Scalar) *Scalar { return a.Mul(b.Pow(-1)) }

// Exp: d(eᵃ)/da = eᵃ — the derivative equals the output itself.
func (a *Scalar) Exp() *Scalar {
	out := newScalar(math.Exp(a.Data), "exp", a)
	out.backward = func() {
		a.Grad += out.Data * out.Grad
	}
	return out
}

// Log: d(ln a)/da = 1/a.
func (a *Scalar) Log() *Scalar {
	out := newScalar(math.Log(a.Data), "log", a)
	out.backward = func() {
		a.Grad += out.Grad / a.Data
	}
	return out
}

// Tanh: d(tanh a)/da = 1 − tanh²(a).
func (a *Scalar) Tanh() *Scalar {
	t := math.Tanh(a.Data)
	out := newScalar(t, "tanh", a)
	out.backward = func() {
		a.Grad += (1 - t*t) * out.Grad
	}
	return out
}

// ReLU: max(0, a); the derivative is 1 for a > 0 and 0 otherwise.
func (a *Scalar) ReLU() *Scalar {
	out := newScalar(math.Max(0, a.Data), "relu", a)
	out.backward = func() {
		if a.Data > 0 {
			a.Grad += out.Grad
		}
	}
	return out
}

// Backward computes dS/dx for every node x the scalar s depends on.
// Nodes are visited in reverse topological order, so a node's Grad is complete
// (all paths summed) before it is pushed further back.
func (s *Scalar) Backward() {
	var order []*Scalar
	seen := map[*Scalar]bool{}
	var visit func(n *Scalar)
	visit = func(n *Scalar) {
		if seen[n] {
			return
		}
		seen[n] = true
		for _, p := range n.prev {
			visit(p)
		}
		order = append(order, n)
	}
	visit(s)

	for _, n := range order {
		n.Grad = 0
	}
	s.Grad = 1
	for i := len(order) - 1; i >= 0; i-- {
		if order[i].backward != nil {
			order[i].backward()
		}
	}
}
