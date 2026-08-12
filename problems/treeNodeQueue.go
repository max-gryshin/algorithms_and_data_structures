package problems

type TreeNodeQ struct {
	*TreeNode
	depth int
}

type TreeNodeQueue []*TreeNodeQ

func NewTreeNodeQueue() TreeNodeQueue {
	return make(TreeNodeQueue, 0)
}
func (t *TreeNodeQueue) Push(node *TreeNode, depth int) {
	*t = append(*t, &TreeNodeQ{TreeNode: node, depth: depth})
}

func (t *TreeNodeQueue) Pop() *TreeNodeQ {
	firstElement := (*t)[0]
	(*t)[0] = nil
	*t = (*t)[1:]

	return firstElement
}
