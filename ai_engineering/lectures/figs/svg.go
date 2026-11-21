package main

import (
	"fmt"
	"html"
	"math"
	"slices"
	"strings"
)

// Palette shared with the other L03 figures (img/L03_partial_*.svg).
const (
	blue     = "#2563eb"
	orange   = "#d97706"
	red      = "#dc2626"
	ink      = "#111827"
	muted    = "#6b7280"
	gridLine = "#e5e7eb"
	cellLine = "#9ca3af"
)

// pen describes how a shape is stroked and filled; zero fields mean "no fill", "no dash", "square corners".
type pen struct {
	color string
	width float64
	fill  string
	rx    float64
	dash  bool
}

func (p pen) attrs() string {
	fill := p.fill
	if fill == "" {
		fill = "none"
	}
	s := fmt.Sprintf(`fill="%s" stroke="%s" stroke-width="%.1f"`, fill, p.color, p.width)
	if p.dash {
		s += ` stroke-dasharray="5 4"`
	}
	return s
}

// font describes a text label; zero fields mean 13px, ink color, left-aligned, normal weight.
type font struct {
	size   float64
	color  string
	anchor string
	bold   bool
}

func (f font) attrs() string {
	if f.size == 0 {
		f.size = 13
	}
	if f.color == "" {
		f.color = ink
	}
	if f.anchor == "" {
		f.anchor = "start"
	}
	weight := "normal"
	if f.bold {
		weight = "bold"
	}
	// The white halo keeps labels readable where they cross lines.
	return fmt.Sprintf(`fill="%s" text-anchor="%s" font-size="%g" font-weight="%s" stroke="white" stroke-width="3" stroke-linejoin="round" paint-order="stroke"`,
		f.color, f.anchor, f.size, weight)
}

// span is a piece of a multi-colored text line.
type span struct{ text, color string }

// canvas accumulates SVG elements; arrow markers are collected per color and emitted in <defs>.
type canvas struct {
	w, h    int
	body    strings.Builder
	markers map[string]bool
}

func newCanvas(w, h int) *canvas { return &canvas{w: w, h: h, markers: map[string]bool{}} }

func (c *canvas) emit(format string, a ...any) {
	fmt.Fprintf(&c.body, format, a...)
	c.body.WriteByte('\n')
}

func (c *canvas) rect(x, y, w, h float64, p pen) {
	c.emit(`<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="%.1f" %s/>`, x, y, w, h, p.rx, p.attrs())
}

func (c *canvas) line(x1, y1, x2, y2 float64, p pen) {
	c.emit(`<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" %s/>`, x1, y1, x2, y2, p.attrs())
}

func (c *canvas) marker(color string) string {
	c.markers[color] = true
	return fmt.Sprintf(` marker-end="url(#a%s)"`, strings.TrimPrefix(color, "#"))
}

func (c *canvas) arrow(x1, y1, x2, y2 float64, p pen) {
	c.emit(`<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" %s%s/>`, x1, y1, x2, y2, p.attrs(), c.marker(p.color))
}

// curveArrow draws a quadratic Bézier from (x1, y1) to (x2, y2) bent towards (cx, cy).
func (c *canvas) curveArrow(x1, y1, cx, cy, x2, y2 float64, p pen) {
	c.emit(`<path d="M%.1f,%.1f Q%.1f,%.1f %.1f,%.1f" %s%s/>`, x1, y1, cx, cy, x2, y2, p.attrs(), c.marker(p.color))
}

func (c *canvas) polyline(pts [][2]float64, p pen) {
	var b strings.Builder
	for _, pt := range pts {
		fmt.Fprintf(&b, "%.1f,%.1f ", pt[0], pt[1])
	}
	c.emit(`<polyline points="%s" %s stroke-linejoin="round"/>`, strings.TrimSpace(b.String()), p.attrs())
}

func (c *canvas) dot(x, y, r float64, color string) {
	c.emit(`<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s" stroke="white" stroke-width="1.5"/>`, x, y, r, color)
}

func (c *canvas) text(x, y float64, s string, f font) {
	c.emit(`<text x="%.1f" y="%.1f" %s>%s</text>`, x, y, f.attrs(), html.EscapeString(s))
}

// textRotated draws a label turned by deg degrees around its anchor point.
func (c *canvas) textRotated(x, y, deg float64, s string, f font) {
	c.emit(`<text x="%.1f" y="%.1f" transform="rotate(%g %.1f %.1f)" %s>%s</text>`, x, y, deg, x, y, f.attrs(), html.EscapeString(s))
}

// spans draws one line of text whose pieces have their own colors.
func (c *canvas) spans(x, y float64, f font, parts ...span) {
	var b strings.Builder
	for _, p := range parts {
		color := p.color
		if color == "" {
			color = ink
		}
		fmt.Fprintf(&b, `<tspan fill="%s">%s</tspan>`, color, html.EscapeString(p.text))
	}
	c.emit(`<text x="%.1f" y="%.1f" %s>%s</text>`, x, y, f.attrs(), b.String())
}

// matrix draws vals as a grid of cells of size s with the top-left corner at (x, y);
// fill picks a background per cell ("" keeps white).
func (c *canvas) matrix(x, y, s float64, vals [][]float64, fill func(i, j int) string) {
	for i, row := range vals {
		for j, v := range row {
			bg := "white"
			if fill != nil {
				if f := fill(i, j); f != "" {
					bg = f
				}
			}
			cx, cy := x+float64(j)*s, y+float64(i)*s
			c.rect(cx, cy, s, s, pen{color: cellLine, width: 1, fill: bg})
			c.text(cx+s/2, cy+s/2+5, fmt.Sprintf("%g", v), font{anchor: "middle", size: 14})
		}
	}
}

func (c *canvas) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, `<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" font-family="Helvetica, Arial, sans-serif" font-size="13">`+"\n", c.w, c.h, c.w, c.h)
	b.WriteString("<defs>")
	colors := make([]string, 0, len(c.markers))
	for color := range c.markers {
		colors = append(colors, color)
	}
	slices.Sort(colors)
	for _, color := range colors {
		fmt.Fprintf(&b, `<marker id="a%s" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="%s"/></marker>`,
			strings.TrimPrefix(color, "#"), color)
	}
	b.WriteString("</defs>\n")
	b.WriteString(`<rect width="100%" height="100%" fill="white"/>` + "\n")
	b.WriteString(c.body.String())
	b.WriteString("</svg>\n")
	return b.String()
}

// minus formats v with sign and a typographic minus: +0.242, −0.341.
func minus(format string, v float64) string {
	return strings.ReplaceAll(fmt.Sprintf(format, v), "-", "−")
}

// pow10 renders 10^k as "10⁻³"; k = 0 and k = 1 give "1" and "10".
func pow10(k int) string {
	switch k {
	case 0:
		return "1"
	case 1:
		return "10"
	}
	sup := strings.NewReplacer("-", "⁻", "0", "⁰", "1", "¹", "2", "²", "3", "³", "4", "⁴", "5", "⁵", "6", "⁶", "7", "⁷", "8", "⁸", "9", "⁹")
	return "10" + sup.Replace(fmt.Sprint(k))
}

// sci renders v as "3·10⁻¹¹" (one significant digit).
func sci(v float64) string {
	e := int(math.Floor(math.Log10(v)))
	m := math.Round(v / math.Pow(10, float64(e)))
	if m == 10 {
		m, e = 1, e+1
	}
	if m == 1 {
		return pow10(e)
	}
	return fmt.Sprintf("%g·%s", m, pow10(e))
}
