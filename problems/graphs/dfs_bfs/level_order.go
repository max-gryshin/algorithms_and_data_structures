package dfs_bfs

import "algorithms_and_data_structures/data_structure"

// 102. Binary Tree Level Order Traversal
// Given the root of a binary tree,
// return the level order traversal of its nodes' values.
// (i.e., from left to right, level by level).

//			   [10]
//			 /	   \
//		   [7]	   [12]
//	      /	 \        \
//	    [6]	  [8]     [14]
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

func levelOrderBst(tree *data_structure.Bst) [][]int {
	if tree == nil {
		// Nothing to do. Can return an emtpy slice of slice of ints to the parent function
		return [][]int{}
	}

	// queue and insert root
	queue := data_structure.NewNodeQueue()
	queue.Push(tree.Root)

	// result slice
	result := make([][]int, 0)

	// Process as long as queue is not empty
	for len(queue) > 0 {
		// Get the current size or length of the queue.
		// This indicates the total number of nodes that are part of current level
		currentSize := len(queue)
		level := make([]int, 0)
		for i := 0; i < currentSize; i++ {
			// take and remove a node
			node := queue.Pop()

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
		// Insert level into the final result
		result = append(result, level)
	}
	return result
}

func levelOrderBstV1(tree *data_structure.Bst) [][]int {
	res := make([][]int, 0)

	queue := []*data_structure.Node{tree.Root}
	for len(queue) > 0 {
		levelSlice := make([]int, 0)
		currentSize := len(queue)
		for i := 0; i < currentSize; i++ {
			el := queue[0]
			queue = queue[1:]
			levelSlice = append(levelSlice, el.Val)
			if el.Left != nil {
				queue = append(queue, el.Left)
			}
			if el.Right != nil {
				queue = append(queue, el.Right)
			}
		}
		res = append(res, levelSlice)
	}
	return res
}
