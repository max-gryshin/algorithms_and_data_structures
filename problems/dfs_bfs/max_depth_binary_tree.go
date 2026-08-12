package dfs_bfs

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func MaxDepthBinaryTree(node *TreeNode) int {
	return depth(node)
}

// DFS - Depth first search
func depth(node *TreeNode) int {
	if node == nil {
		return 0
	}

	left := depth(node.Left)
	right := depth(node.Right)

	return 1 + max(left, right)
}
