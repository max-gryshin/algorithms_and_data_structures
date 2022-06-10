package rsa

import (
	"math/big"
)

const e = 3
const k = 2

type RSA struct {
	p       big.Int
	q       big.Int
	f       *big.Int
	n       *big.Int
	e       *big.Int
	k       big.Int
	private *big.Int
}

func NewRSA(p big.Int, q big.Int) *RSA {
	np := p
	nq := q
	fp := p
	fq := q
	n := p.Mul(&np, &nq)
	f := fp.Sub(&fp, big.NewInt(1)).Mul(&fp, fq.Sub(&fq, big.NewInt(1)))
	priv := *big.NewInt(k)

	return &RSA{
		p:       p,
		q:       q,
		n:       n,
		f:       f,
		e:       big.NewInt(e),
		k:       *big.NewInt(k),
		private: priv.Mul(&priv, f).Add(&priv, big.NewInt(1)).Div(&priv, big.NewInt(e)),
	}
}

func (r *RSA) F() *big.Int {
	return r.f
}

func (r *RSA) N() *big.Int {
	return r.n
}

func (r *RSA) Pub() (*big.Int, *big.Int) {
	return r.e, r.N()
}

func (r *RSA) Private() *big.Int {
	return r.private
}

func (r *RSA) Encrypt(m big.Int) *big.Int {
	return m.Exp(&m, r.e, r.N())
}

func (r *RSA) Decrypt(c *big.Int) *big.Int {
	return c.Exp(c, r.private, r.N())
}
