package weights

import (
	"math"

	"algorithms_and_data_structures/ai_engineering/gpt"
)

// FakeQuantize replaces every 2-D weight matrix of m (Linear W, embeddings, head) by
// its quantize→dequantize round trip, in place, and returns the bytes the matrices
// would take in format f versus float32. The model keeps computing in float32, but
// with exactly the rounding errors of the quantized weights — the standard way to
// measure the quality loss (Δ perplexity) before writing fast kernels.
func FakeQuantize(m *gpt.Model, f Format) (qBytes, f32Bytes int) {
	for _, np := range m.NamedParameters() {
		s := np.P.Data.Shape()
		if len(s) != 2 {
			continue // norms and biases stay float32: tiny and sensitive
		}
		w := np.P.Data.Data()
		var out, in int
		var wt []float32
		if np.Name == "tok.W" || np.Name == "pos.W" { // (rows, d): rows are already "outputs"
			out, in, wt = s[0], s[1], w
		} else { // our Linear is (in, out)
			in, out = s[0], s[1]
			wt = Transpose(w, in, out)
		}
		if f == FormatInt4 && in%32 != 0 {
			continue
		}
		ff := f
		if f == FormatInt4 && (np.Name == "tok.W" || np.Name == "pos.W") {
			// The embedding is also the tied LM head: like llama.cpp (Q6_K "output.weight"
			// in Q4 models) keep it at 8 bits — 4-bit logits blow up the perplexity.
			ff = FormatInt8
		}
		q := Quantize(wt, out, in, ff)
		qBytes += q.Bytes()
		f32Bytes += 4 * len(w)
		d := q.Dequantize()
		if np.Name != "tok.W" && np.Name != "pos.W" {
			d = Transpose(d, out, in)
		}
		copy(w, d)
	}
	return qBytes, f32Bytes
}

// Perplexity is exp(mean next-token cross-entropy) of ids under m, over windows of
// at most m.Cfg.Context tokens.
func Perplexity(m *gpt.Model, ids []int) float64 {
	var nll float64
	n := 0
	for start := 0; start+1 < len(ids); start += m.Cfg.Context {
		end := min(start+m.Cfg.Context+1, len(ids))
		loss := m.Loss([][]int{ids[start : end-1]}, [][]int{ids[start+1 : end]})
		k := end - 1 - start
		nll += float64(loss.Data.Data()[0]) * float64(k)
		n += k
	}
	return math.Exp(nll / float64(n))
}
