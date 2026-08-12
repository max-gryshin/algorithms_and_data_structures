package dfs_bfs

func levelOrder(node *TreeNode) [][]int {
	queue := NewTreeNodeQueue()
	res := make([][]int, 0)
	queue.Push(node, 0)
	length := len(queue)
	for length != 0 {
		element := queue.Pop()
		res[element.depth] = append(res[element.depth], element.Val)
		if element.Right != nil {
			queue.Push(element.Right, element.depth+1)
		}
		if element.Left != nil {
			queue.Push(element.Left, element.depth+1)
		}
		length = len(queue)
	}

	return res
}

// LeetCode
func levelOrderV2(root *TreeNode) [][]int {
	if root == nil {
		// Nothing to do. Can return an emtpy slice of slice of ints to the parent function
		return [][]int{}
	}

	// Create a queue and insert root
	queue := []*TreeNode{}
	queue = append(queue, root)

	// Create result slice
	result := [][]int{}

	// Process as long as queue is not empty
	for len(queue) > 0 {
		// Get the current size or length of the queue.
		// This indicates the total number of nodes that are part of current level
		sz := len(queue)
		level := []int{}
		for i := 0; i < sz; i++ {
			// Remove a node
			node := queue[0]
			queue = queue[1:]

			// Visit the node. Here visiting means collecting it into the output array
			level = append(level, node.Val)

			// Insert children of the node into the queue
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
		// level is filled with one level of nodes' values. Insert this into the final
		// result
		result = append(result, level)
	}
	// result is ready to be returned
	return result
}

//          20
//         /  \
//       10    30
//      / \      \
//     5  15      40
