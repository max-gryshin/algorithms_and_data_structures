package tensor

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestStridesAndAt(t *testing.T) {
	// 2×3 matrix [[1 2 3] [4 5 6]] stored as 1 2 3 4 5 6.
	m := FromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	if got := m.Strides(); got[0] != 3 || got[1] != 1 {
		t.Fatalf("strides = %v, want [3 1]", got)
	}
	if m.At(1, 2) != 6 {
		t.Fatalf("At(1,2) = %v, want 6", m.At(1, 2))
	}
}

func TestTransposeIsView(t *testing.T) {
	m := FromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	tr := m.T()
	if s := tr.Shape(); s[0] != 3 || s[1] != 2 {
		t.Fatalf("shape = %v, want [3 2]", s)
	}
	if tr.At(2, 1) != 6 || tr.At(0, 1) != 4 {
		t.Fatalf("wrong transposed values: %v", tr.Data())
	}
	if tr.IsContiguous() {
		t.Fatal("transpose must not be contiguous")
	}
	tr.Set(100, 0, 1) // writes through the view
	if m.At(1, 0) != 100 {
		t.Fatal("transpose must share memory with the original")
	}
}

func TestReshapeInfer(t *testing.T) {
	x := New(2, 3, 4)
	r := x.Reshape(6, -1)
	if s := r.Shape(); s[0] != 6 || s[1] != 4 {
		t.Fatalf("shape = %v, want [6 4]", s)
	}
}

func TestReshapeOfTransposeCopies(t *testing.T) {
	m := FromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	r := m.T().Reshape(6)
	want := []float32{1, 4, 2, 5, 3, 6}
	for i, v := range r.Data() {
		if v != want[i] {
			t.Fatalf("got %v, want %v", r.Data(), want)
		}
	}
}

func TestBroadcastAdd(t *testing.T) {
	// (2,3) + (3,) : the bias row is added to every row.
	x := FromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	bias := FromSlice([]float32{10, 20, 30}, 3)
	got := Add(x, bias).Data()
	want := []float32{11, 22, 33, 14, 25, 36}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
	if _, err := BroadcastShapes([]int{2, 3}, []int{2}); err == nil {
		t.Fatal("(2,3) and (2,) must not broadcast")
	}
}

func TestMatMulSmall(t *testing.T) {
	// The worked example from the lecture: Q·Kᵀ for the "it/works" sentence.
	q := FromSlice([]float32{0, 1, 2, 0}, 2, 2)
	k := FromSlice([]float32{1, 0, 0, 2}, 2, 2)
	s := MatMul(q, k.T())
	want := []float32{0, 2, 2, 0}
	for i, v := range s.Data() {
		if v != want[i] {
			t.Fatalf("QKᵀ = %v, want %v", s.Data(), want)
		}
	}
}

func TestMatMulKernelsAgree(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for _, sz := range [][3]int{{1, 1, 1}, {3, 5, 7}, {65, 130, 67}, {200, 64, 129}} {
		n, k, p := sz[0], sz[1], sz[2]
		a, b := Randn(rng, n, k).Data(), Randn(rng, k, p).Data()
		ref := make([]float32, n*p)
		MatMulNaive(a, b, ref, n, k, p)
		for name, f := range map[string]func(a, b, out []float32, n, k, p int){
			"naive4": MatMulNaive4, "ikj": MatMulIKJ, "tiled": MatMulTiled, "parallel": MatMulParallel,
		} {
			out := make([]float32, n*p)
			f(a, b, out, n, k, p)
			if !AllClose(FromSlice(out, n, p), FromSlice(ref, n, p), 1e-4, 1e-4) {
				t.Fatalf("%s differs from naive for %v", name, sz)
			}
		}
	}
}

func TestMatMulProperties(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	a, b, c := Randn(rng, 4, 5), Randn(rng, 5, 6), Randn(rng, 6, 3)
	// (AB)ᵀ = BᵀAᵀ
	if !AllClose(MatMul(a, b).T(), MatMul(b.T(), a.T()), 1e-5, 1e-5) {
		t.Fatal("(AB)ᵀ != BᵀAᵀ")
	}
	// (AB)C = A(BC) up to float rounding
	if !AllClose(MatMul(MatMul(a, b), c), MatMul(a, MatMul(b, c)), 1e-4, 1e-4) {
		t.Fatal("matmul is not associative")
	}
}

func TestBatchMatMulBroadcast(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	x := Randn(rng, 2, 3, 4) // (B, T, C)
	w := Randn(rng, 4, 5)    // (C, D) shared by every batch element
	y := MatMul(x, w)
	if s := y.Shape(); s[0] != 2 || s[1] != 3 || s[2] != 5 {
		t.Fatalf("shape = %v, want [2 3 5]", s)
	}
	for b := range 2 {
		xb := FromSlice(append([]float32(nil), x.Data()[b*12:(b+1)*12]...), 3, 4)
		yb := FromSlice(append([]float32(nil), y.Data()[b*15:(b+1)*15]...), 3, 5)
		if !AllClose(yb, MatMul(xb, w), 1e-5, 1e-5) {
			t.Fatalf("batch %d differs", b)
		}
	}
}

// TestDotProductVariance checks the claim behind the 1/√d_k factor in attention:
// for q, k ~ N(0, 1) the dot product q·k has variance d, and q·k/√d has variance 1.
func TestDotProductVariance(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	const samples = 20000
	for _, d := range []int{4, 64, 512} {
		var sum, sumSq, sumScaled, sumScaledSq float64
		for range samples {
			s := float64(Dot(Randn(rng, d), Randn(rng, d)))
			sc := s / math.Sqrt(float64(d))
			sum, sumSq = sum+s, sumSq+s*s
			sumScaled, sumScaledSq = sumScaled+sc, sumScaledSq+sc*sc
		}
		variance := sumSq/samples - (sum/samples)*(sum/samples)
		scaled := sumScaledSq/samples - (sumScaled/samples)*(sumScaled/samples)
		if math.Abs(variance/float64(d)-1) > 0.05 || math.Abs(scaled-1) > 0.05 {
			t.Fatalf("d=%d: Var(q·k)=%.1f (want ≈%d), Var(q·k/√d)=%.3f (want ≈1)", d, variance, d, scaled)
		}
	}
}
