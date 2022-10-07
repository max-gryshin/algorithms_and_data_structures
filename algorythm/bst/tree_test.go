package bst

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

var bstData = []struct {
	n        int
	expected *BST
}{
	{
		n:        int(math.Pow(2, 20)), // 1 million
		expected: NewBST(),
	},
}

func TestBST_Same(t *testing.T) {
	bst1 := NewBST()
	bst2 := NewBST()
	rand.Seed(time.Now().Unix())
	for _, v := range bstData {
		for _, tn := range rand.Perm(v.n) {
			bst1.Insert(tn)
		}
		for _, tn := range rand.Perm(v.n) {
			bst2.Insert(tn)
		}
	}
	if !bst1.Same(bst2) {
		t.Errorf("error %v, %v", bst1, bst2)
	}
}
