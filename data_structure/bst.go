package data_structure

// Bst - Binary search tree
type Bst struct {
	Root *Node
}

type Node struct {
	Val   int
	Left  *Node
	Right *Node
}

func NewBst() *Bst {
	return &Bst{}
}

func (b *Bst) Add(val int) {
	if b.Root == nil {
		b.Root = &Node{Val: val}
		return
	}

	walkAndPut(b.Root, val)
}

func walkAndPut(node *Node, val int) {
	if val <= node.Val {
		if node.Left != nil {
			walkAndPut(node.Left, val)
			return
		}
		node.Left = &Node{Val: val}
		return
	}
	if node.Right != nil {
		walkAndPut(node.Right, val)
		return
	}
	node.Right = &Node{Val: val}
}

func (b *Bst) MaxDepth() int {
	if b.Root == nil {
		return 0
	}
	return depth(b.Root)
}

func depth(node *Node) int {
	if node == nil {
		return 0
	}

	left := depth(node.Left)
	right := depth(node.Right)

	return 1 + max(left, right)
}
