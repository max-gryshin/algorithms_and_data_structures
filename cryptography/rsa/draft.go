// Package rsa contains
// draft to show how rsa works generally
package rsa

import (
	"math/big"
)

const exp = 65537

// Prime number
// e < f
var e = big.NewInt(exp) // public exponent
var bigOne = big.NewInt(1)
var bigOneNeg = big.NewInt(-1)

// RSA simple implementation RSA algorithm
// f = (p-1)*(q-1)
// d = (k*f(n)+1)/e
type RSA struct {
	p               big.Int
	q               big.Int
	f               *big.Int
	n               *big.Int
	privateExponent *big.Int
}

func NewRSA(p big.Int, q big.Int) *RSA {
	np, nq, fp, fq := p, q, p, q
	f := fp.Sub(&fp, bigOne).Mul(&fp, fq.Sub(&fq, bigOne))
	e1 := big.NewInt(exp)

	return &RSA{
		p:               p,
		q:               q,
		n:               p.Mul(&np, &nq),
		f:               f,
		privateExponent: e1.Exp(e1, bigOneNeg, f),
	}
}

func (r *RSA) F() *big.Int {
	return r.f
}

func (r *RSA) N() *big.Int {
	return r.n
}

func (r *RSA) Pub() (*big.Int, *big.Int) {
	return e, r.n
}

func (r *RSA) Private() *big.Int {
	return r.privateExponent
}

// Encrypt given message
// must be less than n
func (r *RSA) Encrypt(m *big.Int) *big.Int {
	return m.Exp(m, e, r.n)
}

func (r *RSA) Decrypt(c *big.Int) *big.Int {
	return c.Exp(c, r.privateExponent, r.n)
}
