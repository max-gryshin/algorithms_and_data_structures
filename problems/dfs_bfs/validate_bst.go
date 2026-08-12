package dfs_bfs

import "math"

//          20
//         /  \
//       10    30
//      / \      \
//     5  15      40

// to validate bst we need walk over all elevement going down to the leaf nodes
// to check is a current node valid we need to pass a valid range to validation function which
// executes recursively
// for the left child node we need to pass range absMin and parent's node value.
// Opposite for the right child node

func isValidBST(root *TreeNode) bool {
	return validate(root, math.MinInt, math.MaxInt)
}

func validate(node *TreeNode, min, max int) bool {
	if node == nil {
		return true
	}
	if min < node.Val && node.Val < max {
		return validate(node.Left, min, node.Val) && validate(node.Right, node.Val, max)
	}

	return false
}
