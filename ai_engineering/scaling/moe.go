package scaling

import (
	"cmp"
	"math"
	"slices"

	"algorithms_and_data_structures/ai_engineering/tensor"
)

// MoE is a forward-only Mixture-of-Experts layer: a router picks the top-k of E experts
// for every token and the output is the router-weighted sum of those experts' outputs.
// Each expert here is a single d×d matrix to keep the routing logic in focus.
type MoE struct {
	Router  *tensor.Tensor   // (d, E): router logits = x · Router
	Experts []*tensor.Tensor // E matrices of shape (d, d)
	K       int              // experts per token
}

// Routing describes the decisions for a batch of T tokens.
type Routing struct {
	Experts [][]int     // (T, K): chosen expert ids per token
	Weights [][]float32 // (T, K): renormalized gate weights per token (sum to 1)
	Load    []float64   // f_e: fraction of token→expert assignments that went to expert e
	Prob    []float64   // P_e: mean router probability of expert e over the batch
	AuxLoss float64     // E · Σ_e f_e · P_e — equals 1 for perfectly uniform routing
}

// Route computes router probabilities, top-k selection and load-balancing statistics for x: (T, d).
func (m *MoE) Route(x *tensor.Tensor) Routing {
	logits := tensor.MatMul(x, m.Router) // (T, E)
	t, e := logits.Shape()[0], logits.Shape()[1]
	r := Routing{
		Experts: make([][]int, t), Weights: make([][]float32, t),
		Load: make([]float64, e), Prob: make([]float64, e),
	}
	for i := range t {
		row := make([]float64, e)
		for j := range e {
			row[j] = float64(logits.At(i, j))
		}
		p := softmax(row)
		for j, v := range p {
			r.Prob[j] += v / float64(t)
		}

		ids := make([]int, e)
		for j := range ids {
			ids[j] = j
		}
		slices.SortStableFunc(ids, func(a, b int) int { return cmp.Compare(p[b], p[a]) })
		top := ids[:m.K]

		var sum float64
		for _, id := range top {
			sum += p[id]
		}
		r.Experts[i] = append([]int(nil), top...)
		r.Weights[i] = make([]float32, m.K)
		for k, id := range top {
			r.Weights[i][k] = float32(p[id] / sum)
			r.Load[id] += 1 / float64(t*m.K)
		}
	}
	for j := range e {
		r.AuxLoss += float64(e) * r.Load[j] * r.Prob[j]
	}
	return r
}

// Forward returns y: (T, d) where y_t = Σ_k w_tk · (x_t · Expert[e_tk]).
// Only K of E experts run per token: compute grows with K, parameters with E.
func (m *MoE) Forward(x *tensor.Tensor) (*tensor.Tensor, Routing) {
	r := m.Route(x)
	t, d := x.Shape()[0], x.Shape()[1]
	out := tensor.New(t, d)
	for i := range t {
		xi := tensor.FromSlice(append([]float32(nil), x.Data()[i*d:(i+1)*d]...), 1, d)
		for k, id := range r.Experts[i] {
			yi := tensor.MatMul(xi, m.Experts[id]).Data()
			for j := range d {
				out.Set(out.At(i, j)+r.Weights[i][k]*yi[j], i, j)
			}
		}
	}
	return out, r
}

func softmax(z []float64) []float64 {
	maxZ := slices.Max(z)
	out := make([]float64, len(z))
	var sum float64
	for i, v := range z {
		out[i] = math.Exp(v - maxZ)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}
