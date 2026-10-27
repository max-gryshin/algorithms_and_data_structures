package autograd

import (
	"fmt"
	"slices"

	"algorithms_and_data_structures/ai_engineering/tensor"
)

// Var is a node of a computational graph over tensors.
//
//	Data — value computed on the forward pass;
//	Grad — dL/dData with the same shape as Data, filled by Backward (nil until then).
//
// Leaves created with Param accumulate gradients across Backward calls
// (that is how gradient accumulation works); call ZeroGrad (or the optimizer's
// ZeroGrad) before each step. Intermediate nodes are rebuilt on every forward pass.
type Var struct {
	Data         *tensor.Tensor
	Grad         *tensor.Tensor
	RequiresGrad bool
	Name         string

	prev     []*Var
	backward func() // reads v.Grad and accumulates into prev[*].Grad
}

// NewVar wraps a tensor as a constant: no gradient is computed for it.
func NewVar(t *tensor.Tensor) *Var { return &Var{Data: t} }

// Param wraps a tensor as a trainable parameter.
func Param(t *tensor.Tensor) *Var { return &Var{Data: t, RequiresGrad: true} }

// Const builds a scalar constant (shape []), handy for x + c or x * c via broadcasting.
func Const(c float32) *Var { return NewVar(tensor.FromSlice([]float32{c})) }

func (v *Var) Shape() []int { return v.Data.Shape() }

// Item returns the value of a single-element Var (e.g. a loss).
func (v *Var) Item() float32 {
	if v.Data.Size() != 1 {
		panic(fmt.Sprintf("autograd: Item on shape %v", v.Data.Shape()))
	}
	return v.Data.Data()[0]
}

// Detach returns a constant that shares Data but cuts the graph: no gradient flows through it.
func (v *Var) Detach() *Var { return NewVar(v.Data) }

// ZeroGrad drops the accumulated gradient.
func (v *Var) ZeroGrad() { v.Grad = nil }

func (v *Var) String() string {
	return fmt.Sprintf("Var(%s %v requiresGrad=%v)", v.Name, v.Data.Shape(), v.RequiresGrad)
}

// newOp creates the output node of an operation. The backward closure is kept
// only if some input needs a gradient — constants cost nothing.
func newOp(data *tensor.Tensor, backward func(out *Var), inputs ...*Var) *Var {
	out := &Var{Data: data}
	for _, in := range inputs {
		if in.RequiresGrad {
			out.RequiresGrad = true
			break
		}
	}
	if out.RequiresGrad {
		out.prev = inputs
		out.backward = func() { backward(out) }
	}
	return out
}

// accum adds g into v.Grad, first summing g over broadcast dimensions so that
// it has v's shape (the gradient of a broadcast is a sum: one input value was
// used in many places, and each use contributes).
func (v *Var) accum(g *tensor.Tensor) {
	if !v.RequiresGrad {
		return
	}
	g = reduceTo(g, v.Data.Shape())
	if v.Grad == nil {
		v.Grad = tensor.FromSlice(slices.Clone(g.Data()), v.Data.Shape()...)
		return
	}
	dst, src := v.Grad.Data(), g.Data()
	for i := range dst {
		dst[i] += src[i]
	}
}

// Backward computes d(v)/d(x) for every x that v depends on. v must hold a single
// element (a loss). Nodes are processed in reverse topological order, so each
// node's gradient is complete before it is propagated further.
func (v *Var) Backward() {
	if v.Data.Size() != 1 {
		panic(fmt.Sprintf("autograd: Backward on non-scalar shape %v", v.Data.Shape()))
	}
	order := topo(v)
	for _, n := range order {
		if n.backward != nil {
			n.Grad = nil // intermediate nodes start clean; leaves keep accumulating
		}
	}
	v.Grad = tensor.FromSlice([]float32{1}, v.Data.Shape()...)
	for i := len(order) - 1; i >= 0; i-- {
		if n := order[i]; n.backward != nil && n.Grad != nil {
			n.backward()
		}
	}
}

func topo(root *Var) []*Var {
	var order []*Var
	seen := map[*Var]bool{}
	// Iterative DFS: graphs of deep models are too deep for comfortable recursion.
	type frame struct {
		v    *Var
		next int
	}
	stack := []frame{{v: root}}
	seen[root] = true
	for len(stack) > 0 {
		top := &stack[len(stack)-1]
		if top.next < len(top.v.prev) {
			p := top.v.prev[top.next]
			top.next++
			if !seen[p] && p.RequiresGrad {
				seen[p] = true
				stack = append(stack, frame{v: p})
			}
			continue
		}
		order = append(order, top.v)
		stack = stack[:len(stack)-1]
	}
	return order
}

// reduceTo sums g over the dimensions that were broadcast to obtain g's shape from shape.
func reduceTo(g *tensor.Tensor, shape []int) *tensor.Tensor {
	gs := g.Shape()
	if slices.Equal(gs, shape) {
		return g
	}
	if len(shape) > 0 && isSuffix(gs, shape) { // bias pattern: sum the rows
		out := make([]float32, numel(shape))
		m := len(out)
		for i, x := range g.Data() {
			out[i%m] += x
		}
		return tensor.FromSlice(out, shape...)
	}
	out := tensor.New(shape...)
	od, gd := out.Data(), g.Data()
	// Strides of out as seen from g's index space: 0 for broadcast/missing dims.
	ostr := make([]int, len(gs))
	shift := len(gs) - len(shape)
	acc := 1
	for d := len(shape) - 1; d >= 0; d-- {
		if shape[d] != 1 {
			ostr[d+shift] = acc
		}
		acc *= shape[d]
	}
	idx := make([]int, len(gs))
	off := 0
	for i := range gd {
		od[off] += gd[i]
		for d := len(gs) - 1; d >= 0; d-- { // odometer increment with running offset
			idx[d]++
			off += ostr[d]
			if idx[d] < gs[d] {
				break
			}
			off -= idx[d] * ostr[d]
			idx[d] = 0
		}
	}
	return out
}

func numel(shape []int) int {
	n := 1
	for _, s := range shape {
		n *= s
	}
	return n
}
