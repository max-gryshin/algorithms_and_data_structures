package agent

import (
	"context"
	"fmt"
	"math"
	"strconv"
	"unicode"
)

// Eval evaluates an arithmetic expression with a recursive-descent parser:
//
//	expr   = term { ("+" | "-") term }
//	term   = unary { ("*" | "/" | "%") unary }
//	unary  = "-" unary | power
//	power  = atom [ "^" unary ]            (right-associative: 2^3^2 = 2^9)
//	atom   = number | name | name "(" expr ")" | "(" expr ")"
//
// Names: pi, e; functions: sqrt, abs, ln, log2, log10, exp.
// Models are bad at long arithmetic (digits are split into tokens unevenly, L5),
// so a calculator is the classic first tool.
func Eval(expr string) (float64, error) {
	p := &parser{src: []rune(expr)}
	v, err := p.expr()
	if err != nil {
		return 0, err
	}
	p.skip()
	if p.pos < len(p.src) {
		return 0, fmt.Errorf("unexpected %q at position %d", p.src[p.pos], p.pos)
	}
	if math.IsInf(v, 0) || math.IsNaN(v) {
		return 0, fmt.Errorf("result is not a finite number")
	}
	return v, nil
}

type parser struct {
	src []rune
	pos int
}

func (p *parser) skip() {
	for p.pos < len(p.src) && unicode.IsSpace(p.src[p.pos]) {
		p.pos++
	}
}

func (p *parser) peek() rune {
	p.skip()
	if p.pos < len(p.src) {
		return p.src[p.pos]
	}
	return 0
}

func (p *parser) expr() (float64, error) {
	v, err := p.term()
	for err == nil {
		op := p.peek()
		if op != '+' && op != '-' {
			break
		}
		p.pos++
		var r float64
		if r, err = p.term(); op == '+' {
			v += r
		} else {
			v -= r
		}
	}
	return v, err
}

func (p *parser) term() (float64, error) {
	v, err := p.unary()
	for err == nil {
		op := p.peek()
		if op != '*' && op != '/' && op != '%' {
			break
		}
		p.pos++
		var r float64
		if r, err = p.unary(); err != nil {
			break
		}
		switch {
		case op == '*':
			v *= r
		case r == 0:
			return 0, fmt.Errorf("division by zero")
		case op == '/':
			v /= r
		default:
			v = math.Mod(v, r)
		}
	}
	return v, err
}

func (p *parser) unary() (float64, error) {
	if p.peek() == '-' {
		p.pos++
		v, err := p.unary()
		return -v, err
	}
	return p.power()
}

func (p *parser) power() (float64, error) {
	base, err := p.atom()
	if err != nil || p.peek() != '^' {
		return base, err
	}
	p.pos++
	exp, err := p.unary()
	return math.Pow(base, exp), err
}

var calcFuncs = map[string]func(float64) float64{
	"sqrt": math.Sqrt, "abs": math.Abs, "ln": math.Log, "log2": math.Log2, "log10": math.Log10, "exp": math.Exp,
}

func (p *parser) atom() (float64, error) {
	c := p.peek()
	switch {
	case c == '(':
		p.pos++
		v, err := p.expr()
		if err != nil {
			return 0, err
		}
		if p.peek() != ')' {
			return 0, fmt.Errorf("missing ')' at position %d", p.pos)
		}
		p.pos++
		return v, nil
	case unicode.IsDigit(c) || c == '.':
		start := p.pos
		for p.pos < len(p.src) && (unicode.IsDigit(p.src[p.pos]) || p.src[p.pos] == '.' ||
			p.src[p.pos] == 'e' && p.pos+1 < len(p.src) && (unicode.IsDigit(p.src[p.pos+1]) || p.src[p.pos+1] == '-')) {
			if p.src[p.pos] == 'e' {
				p.pos++ // exponent: 1e-3
			}
			p.pos++
		}
		return strconv.ParseFloat(string(p.src[start:p.pos]), 64)
	case unicode.IsLetter(c):
		start := p.pos
		for p.pos < len(p.src) && (unicode.IsLetter(p.src[p.pos]) || unicode.IsDigit(p.src[p.pos])) {
			p.pos++
		}
		name := string(p.src[start:p.pos])
		switch name {
		case "pi":
			return math.Pi, nil
		case "e":
			return math.E, nil
		}
		f, ok := calcFuncs[name]
		if !ok || p.peek() != '(' {
			return 0, fmt.Errorf("unknown name %q (known: pi, e, sqrt, abs, ln, log2, log10, exp)", name)
		}
		v, err := p.atom() // parenthesized argument
		return f(v), err
	case c == 0:
		return 0, fmt.Errorf("unexpected end of expression")
	}
	return 0, fmt.Errorf("unexpected %q at position %d", c, p.pos)
}

// CalculatorInput is the input of the calculator tool.
type CalculatorInput struct {
	Expression string `json:"expression" desc:"Arithmetic expression, e.g. \"(17*23 + 2^10) / 3\". Operators: + - * / % ^ and parentheses; functions sqrt, abs, ln, log2, log10, exp; constants pi, e."`
}

// Calculator returns the calculator tool.
func Calculator() Tool {
	return NewTool("calculator",
		"Evaluate an arithmetic expression exactly. Use it for ANY arithmetic beyond single-digit mental math "+
			"instead of computing in your head. Returns the numeric result.",
		func(_ context.Context, in CalculatorInput) (string, error) {
			v, err := Eval(in.Expression)
			if err != nil {
				return "", err
			}
			return strconv.FormatFloat(v, 'g', -1, 64), nil
		})
}
