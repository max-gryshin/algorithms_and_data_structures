package dfs_bfs

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

//          20
//         /  \
//       10    30
//      / \      \
//     5  15      40
//					\
//					 50

func MaxDepthBinaryTree(node *TreeNode) int {
	return depth(node)
}

// DFS - Depth first search
func depth(node *TreeNode) int {
	if node == nil {
		return 0
	}

	return 1 + max(depth(node.Left), depth(node.Right))
}
