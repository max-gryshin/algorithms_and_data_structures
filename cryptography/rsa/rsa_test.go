package rsa

import (
	"math/big"
	"testing"
)

func TestRsa(t *testing.T) {
	p := big.NewInt(17)
	q := big.NewInt(23)
	rsa := NewRSA(*p, *q)
	m := big.NewInt(32)
	c := rsa.Encrypt(m)
	res := rsa.Decrypt(c)
	if m != res {
		t.Errorf(
			" Result of Decrypt(%v) not correct: expected %d, actual %d",
			c,
			m,
			res,
		)
	}
}
