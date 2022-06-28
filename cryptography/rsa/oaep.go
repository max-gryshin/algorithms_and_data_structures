// Package rsa contains
// Optimal Asymmetric Encryption Padding.
// is a padding scheme often used together with rsa encryption.
// It takes huge of work and require perfect knowledge of
// cryptographic primitives
// Could be implemented later
package rsa

import (
	"hash"
	"math/big"
)

type Public struct {
	n *big.Int
	e int // public exponent 65537
}

type Private struct {
	n      *big.Int
	d      *big.Int   // private exponent
	Primes []*big.Int // prime factors of N
}

func NewPrivate(n *big.Int, d *big.Int, nprimes int) *Private {
	return &Private{
		n:      n,
		d:      d,
		Primes: make([]*big.Int, nprimes),
	}
}

func EncryptOAEP(h hash.Hash, pub *Public, M []byte) ([]byte, error) {
	return nil, nil
}
