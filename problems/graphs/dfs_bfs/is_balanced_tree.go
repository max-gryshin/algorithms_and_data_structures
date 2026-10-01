package dfs_bfs

import "algorithms_and_data_structures/data_structure"

//	     1
//	    / \
//	   2   3
//	  /
//	 4
//	/
// 5

func isBalanced(root *data_structure.Node) bool {
	if root == nil {
		return true
	}
	return dfsIsBalanced(root) != -1
}

func dfsIsBalanced(node *data_structure.Node) int {
	if node == nil {
		return 0
	}
	left := dfsIsBalanced(node.Left)
	if left == -1 {
		return -1
	}
	right := dfsIsBalanced(node.Right)
	if right == -1 {
		return -1
	}
	heightDiff := left - right
	if heightDiff < 0 {
		heightDiff *= -1
	}
	if heightDiff > 1 {
		return -1
	}
	return max(left, right) + 1
}
