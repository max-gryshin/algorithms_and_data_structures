package bst

import "fmt"

type BST struct {
	root *TreeNode
}

type TreeNode struct {
	val    int
	left   *TreeNode
	right  *TreeNode
	parent *TreeNode
}

func NewBST() *BST {
	return &BST{}
}

func (t *BST) Insert(z int) {
	tn := TreeNode{val: z}
	var y *TreeNode
	x := t.root
	for x != nil {
		y = x
		if tn.val < x.val {
			x = x.left
		} else {
			x = x.right
		}
	}
	tn.parent = y
	if y == nil {
		t.root = &tn
	} else if tn.val < y.val {
		y.left = &tn
	} else {
		y.right = &tn
	}
}

func (t *BST) Search(i int) *TreeNode {
	x := t.root
	if x.val == i {
		return x
	}
	for x != nil && i != x.val {
		if x.val < i {
			x = x.left
		} else {
			x = x.right
		}
	}
	return x
}

func Min(x *TreeNode) *TreeNode {
	for x.left != nil {
		x = x.left
	}
	return x
}

func Max(x *TreeNode) *TreeNode {
	fmt.Printf("MAx() val %v", x)
	for x.left != nil {
		x = x.right
	}
	return x
}

func (t *BST) Transplant(u, v *TreeNode) {
	if u.parent == nil {
		t.root = v
	} else if u == u.parent.left {
		u.parent.left = v
	} else {
		u.parent.right = v
	}
	if v != nil {
		v.parent = u.parent
	}
}

func (t *BST) Delete(z *TreeNode) {
	var y *TreeNode
	if z.left == nil {
		t.Transplant(z, z.right)
	} else if z.right == nil {
		t.Transplant(z, z.left)
	} else {
		y = Min(z.right)
		if y.parent != z {
			t.Transplant(y, y.right)
			y.right = z.right
			y.right.parent = y
		}
		t.Transplant(z, y)
		y.left = z.left
		y.left.parent = y
	}
}

func (t *BST) Walk(ch chan int) {
	defer close(ch)
	if t.root == nil {
		return
	}
	var y *TreeNode
	for t.root != y {
		if t.root == nil {
			break
		}
		y = Min(t.root)
		ch <- y.val
		t.Delete(y)
	}
	for t.root != y {
		if t.root == nil {
			break
		}
		y = Max(t.root)
		ch <- y.val
		t.Delete(y)
	}
}

func (t *BST) Same(bst *BST) bool {
	ch1, ch2 := make(chan int), make(chan int)
	go t.Walk(ch1)
	go bst.Walk(ch2)
	for {
		n1, ok1 := <-ch1
		n2, ok2 := <-ch2
		if n1 != n2 || ok1 != ok2 {
			return false
		}
		if !ok1 {
			break
		}
	}
	return true
}
