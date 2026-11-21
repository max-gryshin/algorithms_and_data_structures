package main

import (
	"fmt"
	"math"

	"algorithms_and_data_structures/ai_engineering/autograd"
	"algorithms_and_data_structures/ai_engineering/tensor"
)

// l03LMHeadGraph: the last layer of a language model as a graph — forward values left to right,
// gradients right to left, labelled with the formulas of section 5.
func l03LMHeadGraph() *canvas {
	c := newCanvas(680, 330)
	c.arrow(20, 22, 52, 22, pen{color: blue, width: 2})
	c.text(60, 26, "forward: значения", font{color: blue})
	c.arrow(220, 22, 252, 22, pen{color: red, width: 2})
	c.text(260, 26, "backward: градиенты", font{color: red})

	const top, h = 130.0, 54.0
	mid := top + h/2
	node := func(x, w float64, title, sub, fill string) {
		c.rect(x, top, w, h, pen{color: muted, width: 1.2, fill: fill, rx: 6})
		if sub == "" {
			c.text(x+w/2, mid+5, title, font{anchor: "middle", bold: true, size: 15})
			return
		}
		c.text(x+w/2, mid-3, title, font{anchor: "middle", bold: true, size: 15})
		c.text(x+w/2, mid+15, sub, font{anchor: "middle", size: 11, color: muted})
	}
	node(20, 110, "X", "активации n×C", "white")
	node(175, 90, "MatMul", "", "#f3f4f6")
	node(310, 110, "Z", "логиты n×V", "white")
	node(465, 120, "softmax → P", "+ CE, среднее по n", "#f3f4f6")
	node(615, 40, "L", "", "white")

	// W enters MatMul from above.
	c.rect(175, 55, 90, 40, pen{color: muted, width: 1.2, fill: "white", rx: 6})
	c.text(220, 73, "W", font{anchor: "middle", bold: true, size: 15})
	c.text(220, 88, "веса C×V", font{anchor: "middle", size: 11, color: muted})

	fwd := pen{color: blue, width: 2}
	c.arrow(130, mid, 175, mid, fwd)
	c.arrow(265, mid, 310, mid, fwd)
	c.arrow(420, mid, 465, mid, fwd)
	c.arrow(585, mid, 615, mid, fwd)
	c.arrow(205, 95, 205, top, fwd)

	bwd := pen{color: red, width: 2}
	c.arrow(235, top, 235, 95, bwd)
	c.text(245, 116, "∂L/∂W = Xᵀ·G   (C×V)", font{color: red})

	bottom := top + h
	c.text(670, 206, "∂L/∂L = 1", font{color: red, size: 11, anchor: "end"})
	c.curveArrow(635, bottom, 495, 262, 365, bottom, bwd)
	c.text(495, 243, "∂L/∂Z = G = (P − Y)/n", font{color: red, anchor: "middle"})
	c.text(495, 260, "форма n×V, как у Z", font{color: muted, anchor: "middle", size: 11})
	c.curveArrow(365, bottom, 220, 262, 75, bottom, bwd)
	c.text(220, 243, "∂L/∂X = G·Wᵀ", font{color: red, anchor: "middle"})
	c.text(220, 260, "n×C → в предыдущий слой", font{color: muted, anchor: "middle", size: 11})

	c.text(20, 298, "Backward начинается с ∂L/∂L = 1 и идёт справа налево: формулы раздела 5 по очереди.", font{size: 12, color: muted})
	c.text(20, 316, "Y — one-hot правильных токенов, n = батч·позиции.", font{size: 12, color: muted})
	return c
}

// l03MatMulGrad: the 2×2 example of section 5 — where the 11 in ∂L/∂A comes from and how
// the formula G·Bᵀ is recovered from shapes. All numbers come from autograd.MatMul.
func l03MatMulGrad() *canvas {
	a := autograd.Param(tensor.FromSlice([]float32{1, 2, 3, 4}, 2, 2))
	b := autograd.Param(tensor.FromSlice([]float32{5, 6, 7, 8}, 2, 2))
	prod := autograd.MatMul(a, b)
	autograd.Sum(prod).Backward()
	A, B, C, dA := rows(a.Data), rows(b.Data), rows(prod.Data), rows(a.Grad)
	G := [][]float64{{1, 1}, {1, 1}}
	Bt := [][]float64{{B[0][0], B[1][0]}, {B[0][1], B[1][1]}}

	const (
		s                   = 40.0
		xA, xB, xC          = 40.0, 170.0, 300.0
		y1, y2              = 90.0, 230.0
		strong, light       = "#fde68a", "#fef3c7"
		strongRed, lightRed = "#fecaca", "#fee2e2"
	)
	c := newCanvas(680, 430)

	// Forward row: a₁₁ feeds c₁₁ (times b₁₁) and c₁₂ (times b₁₂).
	c.matrix(xA, y1, s, A, func(i, j int) string { return pick(i == 0 && j == 0, strong) })
	c.matrix(xB, y1, s, B, func(i, j int) string { return pick(i == 0, light) })
	c.matrix(xC, y1, s, C, func(i, j int) string { return pick(i == 0, light) })
	c.text(150, y1+s+5, "·", font{anchor: "middle", size: 20})
	c.text(282, y1+s+6, "=", font{anchor: "middle", size: 18})
	c.text(xA+s, y1+2*s+22, "A", font{anchor: "middle", bold: true})
	c.text(xB+s, y1+2*s+22, "B", font{anchor: "middle", bold: true})
	c.text(xC+s, y1+2*s+22, "C = A·B", font{anchor: "middle", bold: true})
	arc := pen{color: orange, width: 1.8}
	c.curveArrow(xA+s/2, y1, 190, 20, xC+s/2, y1, arc)
	c.curveArrow(xA+s/2, y1, 210, -6, xC+1.5*s, y1, arc)
	c.text(190, 76, fmt.Sprintf("×b₁₁ = %g", B[0][0]), font{anchor: "middle", color: orange})
	c.text(210, 34, fmt.Sprintf("×b₁₂ = %g", B[0][1]), font{anchor: "middle", color: orange})

	tx := 420.0
	c.text(tx, 100, "Forward", font{bold: true})
	c.text(tx, 122, fmt.Sprintf("a₁₁ входит в c₁₁ (×%g) и в c₁₂ (×%g).", B[0][0], B[0][1]), font{size: 12})
	c.text(tx, 142, "L = сумма C ⇒ G = ∂L/∂C — все единицы.", font{size: 12})
	c.text(tx, 164, fmt.Sprintf("∂L/∂a₁₁ = 1·%g + 1·%g = %g", B[0][0], B[0][1], dA[0][0]), font{bold: true, color: orange})

	// Backward row: the same 11 as (row 1 of G) · (column 1 of Bᵀ).
	c.matrix(xA, y2, s, G, func(i, j int) string { return pick(i == 0, lightRed) })
	c.matrix(xB, y2, s, Bt, func(i, j int) string { return pick(j == 0, lightRed) })
	c.matrix(xC, y2, s, dA, func(i, j int) string { return pick(i == 0 && j == 0, strongRed) })
	c.text(150, y2+s+5, "·", font{anchor: "middle", size: 20})
	c.text(282, y2+s+6, "=", font{anchor: "middle", size: 18})
	c.text(xA+s, y2+2*s+22, "G = ∂L/∂C", font{anchor: "middle", bold: true})
	c.text(xB+s, y2+2*s+22, "Bᵀ", font{anchor: "middle", bold: true})
	c.text(xC+s, y2+2*s+22, "∂L/∂A = G·Bᵀ", font{anchor: "middle", bold: true})

	c.text(tx, 240, "Backward", font{bold: true})
	c.text(tx, 262, "(G·Bᵀ)₁₁ = строка 1 из G", font{size: 12})
	c.text(tx, 280, fmt.Sprintf("· столбец 1 из Bᵀ = 1·%g + 1·%g = %g ✓", Bt[0][0], Bt[1][0], dA[0][0]), font{size: 12, color: red})
	c.text(tx, 302, "Столбец 1 из Bᵀ — это строка 1 из B:", font{size: 12, color: muted})
	c.text(tx, 320, "те же множители, что на стрелках сверху.", font{size: 12, color: muted})

	// Shapes: the inner dimensions (orange) must match and cancel.
	f := font{size: 14}
	c.text(40, 376, "Формы:", f)
	c.spans(100, 376, f, span{"∂L/∂A = G·Bᵀ = (n×", ""}, span{"p", orange}, span{")·(", ""}, span{"p", orange}, span{"×k) = n×k — как у A ✓", ""})
	c.spans(100, 398, f, span{"∂L/∂B = Aᵀ·G = (k×", ""}, span{"n", orange}, span{")·(", ""}, span{"n", orange}, span{"×p) = k×p — как у B ✓", ""})
	c.text(40, 420, "Внутренние размеры (оранжевые) обязаны совпасть — так формулу восстанавливают по формам.", font{size: 12, color: muted})
	return c
}

// l03SoftmaxCEStep: what the gradient p − y does — probabilities before and after one step
// z ← z − (p − y) on the example z = [2, 1, 0.1], correct token 0.
func l03SoftmaxCEStep() *canvas {
	z := []float32{2, 1, 0.1}
	p, g, loss := ceStep(z)
	z2 := make([]float32, len(z))
	for i := range z {
		z2[i] = z[i] - float32(g[i])
	}
	p2, _, loss2 := ceStep(z2)
	y := []float64{1, 0, 0}

	c := newCanvas(680, 380)
	const (
		zero, scale = 270.0, 170.0
		panelW      = 280.0
		barW        = 44.0
	)
	yOf := func(v float64) float64 { return zero - v*scale }
	panel := func(x0 float64, title, sub string, probs, ghost []float64, arrows bool, third func(i int) (string, string)) {
		c.text(x0+panelW/2, 30, title, font{anchor: "middle", bold: true, size: 14})
		c.text(x0+panelW/2, 52, sub, font{anchor: "middle", color: muted, size: 12})
		for _, v := range []float64{0, 0.5, 1} {
			c.line(x0, yOf(v), x0+panelW, yOf(v), pen{color: gridLine, width: 1})
			c.text(x0-6, yOf(v)+4, fmt.Sprintf("%g", v), font{anchor: "end", size: 11, color: muted})
		}
		for i, pv := range probs {
			cx := x0 + panelW*(float64(i)+0.5)/3
			if y[i] > 0 {
				c.rect(cx-barW/2, yOf(y[i]), barW, y[i]*scale, pen{color: muted, width: 1.2, dash: true})
			}
			c.rect(cx-barW/2, yOf(pv), barW, pv*scale, pen{color: blue, width: 1.2, fill: "#bfdbfe"})
			if ghost != nil {
				c.rect(cx-barW/2, yOf(ghost[i]), barW, ghost[i]*scale, pen{color: blue, width: 1.2, dash: true})
			}
			if arrows {
				c.arrow(cx, yOf(pv), cx, yOf(y[i])+2*sign(y[i]-pv), pen{color: red, width: 2})
			}
			name := fmt.Sprintf("токен %d", i)
			if i == 0 {
				name += " · верный"
			}
			c.text(cx, 292, name, font{anchor: "middle", size: 12})
			c.text(cx, 310, fmt.Sprintf("p = %.3f", pv), font{anchor: "middle", size: 12, color: blue})
			text, color := third(i)
			c.text(cx, 328, text, font{anchor: "middle", size: 12, color: color})
		}
	}
	panel(50, "До шага",
		fmt.Sprintf("z = [%g, %g, %g],  L = %.3f", z[0], z[1], z[2], loss),
		p, nil, true,
		func(i int) (string, string) { return "p − y = " + minus("%+.3f", g[i]), red })
	panel(380, "После шага z ← z − (p − y)",
		fmt.Sprintf("z = [%.3f, %.3f, %.3f],  L = %.3f", z2[0], z2[1], z2[2], loss2),
		p2, p, false,
		func(i int) (string, string) { return fmt.Sprintf("было %.3f", p[i]), muted })

	c.rect(50, 350, 14, 14, pen{color: blue, width: 1.2, fill: "#bfdbfe"})
	c.text(70, 362, "p — вероятности модели", font{size: 12})
	c.rect(230, 350, 14, 14, pen{color: muted, width: 1.2, dash: true})
	c.text(250, 362, "y — правильный ответ", font{size: 12})
	c.arrow(410, 364, 410, 348, pen{color: red, width: 2})
	c.text(422, 362, "куда тянет шаг −(p − y)", font{size: 12})
	return c
}

// ceStep returns softmax(z), the gradient p − y of cross-entropy with target 0, and the loss.
func ceStep(z []float32) (p, g []float64, loss float64) {
	v := autograd.Param(tensor.FromSlice(append([]float32(nil), z...), 1, len(z)))
	l := autograd.CrossEntropy(v, []int{0})
	l.Backward()
	return floats(autograd.Softmax(v).Data.Data()), floats(v.Grad.Data()), float64(l.Item())
}

// l03GradCheckEps: relative error of the central difference against the analytic gradient as a
// function of ε, for the cross-entropy example — float32 (autograd.Var + GradCheck, as in the
// tests) and float64 (autograd.Scalar). Truncation error grows as ε², rounding error as 1/ε.
func l03GradCheckEps() *canvas {
	z32 := autograd.Param(tensor.FromSlice([]float32{2, 1, 0.1}, 1, 3))
	f32 := func() *autograd.Var { return autograd.CrossEntropy(z32, []int{0}) }

	type point struct{ eps, rel float64 }
	var c32, c64 []point
	for k := -64; k <= 0; k++ {
		eps := math.Pow(10, float64(k)/8)
		rel32, _ := autograd.GradCheck(f32, []*autograd.Var{z32}, eps)
		c32 = append(c32, point{eps, rel32})
		c64 = append(c64, point{eps, gradCheck64([3]float64{2, 1, 0.1}, eps)})
	}

	const (
		left, right = 80.0, 640.0
		top, bottom = 60.0, 340.0
		decX, decY  = 8.0, 12.0
	)
	xOf := func(eps float64) float64 { return left + (math.Log10(eps)+decX)/decX*(right-left) }
	yOf := func(rel float64) float64 {
		rel = min(max(rel, 1e-12), 1)
		return top - math.Log10(rel)/decY*(bottom-top)
	}

	c := newCanvas(680, 430)
	c.text(340, 24, "Gradient check: ошибка центральной разности в зависимости от ε", font{anchor: "middle", bold: true, size: 14})
	c.text(340, 44, "cross-entropy для z = [2, 1, 0.1]; максимум относительной ошибки по трём логитам", font{anchor: "middle", size: 12, color: muted})
	for k := -8; k <= 0; k++ {
		x := xOf(math.Pow(10, float64(k)))
		c.line(x, top, x, bottom, pen{color: gridLine, width: 1})
		c.text(x, bottom+18, pow10(k), font{anchor: "middle", size: 11, color: muted})
	}
	for k := -12; k <= 0; k += 2 {
		y := yOf(math.Pow(10, float64(k)))
		c.line(left, y, right, y, pen{color: gridLine, width: 1})
		c.text(left-6, y+4, pow10(k), font{anchor: "end", size: 11, color: muted})
	}
	c.rect(left, top, right-left, bottom-top, pen{color: cellLine, width: 1})
	c.text((left+right)/2, bottom+38, "ε — шаг конечной разности", font{anchor: "middle", size: 12})
	c.textRotated(24, (top+bottom)/2, -90, "относительная ошибка rel", font{anchor: "middle", size: 12})

	// Test settings: ε = 10⁻² and the failure threshold of TestGradCheckOps.
	c.line(xOf(1e-2), top, xOf(1e-2), bottom, pen{color: muted, width: 1.2, dash: true})
	c.text(xOf(1e-2)-4, bottom-8, "ε = 10⁻² в тестах", font{anchor: "end", size: 11, color: muted})
	c.line(left, yOf(5e-2), right, yOf(5e-2), pen{color: red, width: 1.2, dash: true})
	c.text(right-4, yOf(5e-2)-6, "порог TestGradCheckOps: 5·10⁻²", font{anchor: "end", size: 11, color: red})

	curve := func(pts []point, color string) point {
		xy := make([][2]float64, len(pts))
		best := pts[0]
		for i, p := range pts {
			xy[i] = [2]float64{xOf(p.eps), yOf(p.rel)}
			if p.rel < best.rel {
				best = p
			}
		}
		c.polyline(xy, pen{color: color, width: 2})
		c.dot(xOf(best.eps), yOf(best.rel), 4.5, color)
		return best
	}
	b32 := curve(c32, orange)
	b64 := curve(c64, blue)
	c.text(xOf(b32.eps)-8, yOf(b32.rel)+18, fmt.Sprintf("минимум: ε ≈ %s, rel ≈ %s", sci(b32.eps), sci(b32.rel)), font{size: 11, color: orange, anchor: "end"})
	c.text(xOf(b64.eps), yOf(b64.rel)+20, fmt.Sprintf("минимум: ε ≈ %s, rel ≈ %s", sci(b64.eps), sci(b64.rel)), font{size: 11, color: blue, anchor: "middle"})

	c.text(xOf(3e-8), yOf(gradCheck64([3]float64{2, 1, 0.1}, 3e-8))-12, "округление ∝ 1/ε", font{size: 12, color: blue})
	c.text(xOf(1.5e-2), yOf(2e-6), "усечение ∝ ε²", font{size: 12, color: blue})

	c.line(150, 410, 178, 410, pen{color: orange, width: 2})
	c.text(184, 414, "float32: autograd.Var + GradCheck (как в тестах)", font{size: 12})
	c.line(470, 410, 498, 410, pen{color: blue, width: 2})
	c.text(504, 414, "float64: autograd.Scalar", font{size: 12})
	return c
}

// gradCheck64 is GradCheck in float64 on Scalar nodes: L = ln Σ e^{zⱼ} − z₀.
func gradCheck64(z [3]float64, eps float64) float64 {
	ce := func(z [3]float64) (float64, [3]float64) {
		n := [3]*autograd.Scalar{autograd.S(z[0]), autograd.S(z[1]), autograd.S(z[2])}
		l := n[0].Exp().Add(n[1].Exp()).Add(n[2].Exp()).Log().Sub(n[0])
		l.Backward()
		return l.Data, [3]float64{n[0].Grad, n[1].Grad, n[2].Grad}
	}
	_, an := ce(z)
	var worst float64
	for i := range z {
		zp, zm := z, z
		zp[i] += eps
		zm[i] -= eps
		lp, _ := ce(zp)
		lm, _ := ce(zm)
		num := (lp - lm) / (2 * eps)
		worst = max(worst, math.Abs(an[i]-num)/math.Max(math.Abs(an[i])+math.Abs(num), 1e-3))
	}
	return worst
}

func rows(t *tensor.Tensor) [][]float64 {
	s := t.Shape()
	out := make([][]float64, s[0])
	for i := range out {
		out[i] = make([]float64, s[1])
		for j := range out[i] {
			out[i][j] = float64(t.At(i, j))
		}
	}
	return out
}

func floats(xs []float32) []float64 {
	out := make([]float64, len(xs))
	for i, x := range xs {
		out[i] = float64(x)
	}
	return out
}

func pick(cond bool, color string) string {
	if cond {
		return color
	}
	return ""
}

func sign(v float64) float64 {
	if v < 0 {
		return -1
	}
	return 1
}
