package dfs_bfs

import "algorithms_and_data_structures/data_structure"

// Given the root of a binary tree and an integer targetSum
// return true if the tree has a root-to-leaf path such that
// adding up all the values along the path equals targetSum.
//
//A leaf is a node with no children.

//Input: root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
//Output: true
//Explanation: The root-to-leaf path with the target sum is shown.

func hasPathSum(root *data_structure.Node, targetSum int) bool {
	if root == nil {
		return false
	}
	return treeSum(root, targetSum, 0)
}

func treeSum(node *data_structure.Node, targetSum int, sum int) bool {
	if node == nil {
		return false
	}
	sum += node.Val
	if node.Left == nil && node.Right == nil {
		if sum == targetSum {
			return true
		}
		return false
	}
	return treeSum(node.Left, targetSum, sum) || treeSum(node.Right, targetSum, sum)
}
