package autograd

import (
	"math"
	"math/rand/v2"
	"testing"
)

// The hand-traced example from L3: L = (a·b + c)·f.
func TestScalarLectureExample(t *testing.T) {
	a, b, c, f := L("a", 2), L("b", -3), L("c", 10), L("f", -2)
	e := a.Mul(b)
	d := e.Add(c)
	loss := d.Mul(f)
	loss.Backward()

	want := map[*Scalar][2]float64{ // node: {data, grad}
		e: {-6, -2}, d: {4, -2}, loss: {-8, 1},
		a: {2, 6}, b: {-3, -4}, c: {10, -2}, f: {-2, 4},
	}
	for n, w := range want {
		if n.Data != w[0] || n.Grad != w[1] {
			t.Errorf("%v: want data=%v grad=%v", n, w[0], w[1])
		}
	}
}

// A node used twice must receive the sum of both contributions: d(x·x)/dx = 2x.
func TestScalarReuseAccumulates(t *testing.T) {
	x := S(3)
	y := x.Mul(x)
	y.Backward()
	if x.Grad != 6 {
		t.Fatalf("grad = %v, want 6", x.Grad)
	}
}

// Every Scalar op is checked against a central finite difference.
func TestScalarOpsFiniteDifference(t *testing.T) {
	ops := map[string]func(x *Scalar) *Scalar{
		"pow3": func(x *Scalar) *Scalar { return x.Pow(3) },
		"exp":  func(x *Scalar) *Scalar { return x.Exp() },
		"log":  func(x *Scalar) *Scalar { return x.Log() },
		"tanh": func(x *Scalar) *Scalar { return x.Tanh() },
		"relu": func(x *Scalar) *Scalar { return x.ReLU() },
		"div":  func(x *Scalar) *Scalar { return S(1).Div(x.Add(S(1))) },
		"mix":  func(x *Scalar) *Scalar { return x.Mul(x).Sub(x.Exp().Mul(S(0.5))).Tanh() },
	}
	const h = 1e-6
	for name, op := range ops {
		for _, x0 := range []float64{0.3, 1.7} {
			x := S(x0)
			op(x).Backward()
			num := (op(S(x0+h)).Data - op(S(x0-h)).Data) / (2 * h)
			if math.Abs(x.Grad-num) > 1e-6 {
				t.Errorf("%s at %v: analytic %v, numeric %v", name, x0, x.Grad, num)
			}
		}
	}
}

// Learn XOR with a 2-4-1 tanh network built from Scalars and plain gradient descent.
func TestScalarXOR(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 1))
	param := func() *Scalar { return S(rng.Float64()*2 - 1) }
	var w1 [4][2]*Scalar
	var b1, w2 [4]*Scalar
	b2 := param()
	var params []*Scalar
	for h := range 4 {
		w1[h] = [2]*Scalar{param(), param()}
		b1[h], w2[h] = param(), param()
		params = append(params, w1[h][0], w1[h][1], b1[h], w2[h])
	}
	params = append(params, b2)

	xs := [4][2]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	ys := [4]float64{-1, 1, 1, -1}
	predict := func(x [2]float64) *Scalar {
		out := b2
		for h := range 4 {
			z := w1[h][0].Mul(S(x[0])).Add(w1[h][1].Mul(S(x[1]))).Add(b1[h]).Tanh()
			out = out.Add(w2[h].Mul(z))
		}
		return out.Tanh()
	}

	var loss *Scalar
	for range 300 {
		loss = S(0)
		for i, x := range xs {
			loss = loss.Add(predict(x).Sub(S(ys[i])).Pow(2))
		}
		loss.Backward()
		for _, p := range params {
			p.Data -= 0.1 * p.Grad
		}
	}
	if loss.Data > 0.05 {
		t.Fatalf("XOR not learned, loss = %v", loss.Data)
	}
	for i, x := range xs {
		if p := predict(x).Data; p*ys[i] <= 0 {
			t.Fatalf("wrong sign for %v: %v", x, p)
		}
	}
}
