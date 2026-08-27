package data_structure

type NodeQueue []*Node

func NewNodeQueue() NodeQueue {
	return make(NodeQueue, 0)
}

func (nq *NodeQueue) Push(newNode *Node) {
	*nq = append(*nq, newNode)
}

func (nq *NodeQueue) Pop() *Node {
	node := (*nq)[0]
	(*nq)[0] = nil
	*nq = (*nq)[1:]

	return node
}
