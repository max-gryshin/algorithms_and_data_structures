package dfs_bfs

import "algorithms_and_data_structures/data_structure"

// Lowest Common Ancestor of a Binary Tree.

//        3
//       / \
//      5   1
//     / \ / \
//    6  2 0  8
//      / \
//     7   4

//   p = 5
//   q = 1
// res = 3

//   p = 5
//   q = 4
// res =5

// left  = DFS(node.Left)
//right = DFS(node.Right)
//
//if left != nil && right != nil:
//    return node
//
//if left != nil:
//    return left
//
//return right

func lcaBtree(root *data_structure.Node, p, q int) *data_structure.Node {
	return dfsLca(root, p, q)
}

// DFS-паттерн:
// 1. base case
// 2. current node is target?
// 3. DFS left
// 4. DFS right
// 5. combine results
func dfsLca(node *data_structure.Node, p, q int) *data_structure.Node {
	if node == nil {
		return nil
	}

	if node.Val == p || node.Val == q {
		return node
	}

	left := dfsLca(node.Left, p, q)
	right := dfsLca(node.Right, p, q)

	if left != nil && right != nil {
		return node
	}

	if left != nil {
		return left
	}

	return right
}

//	   3
//	  / \
//	 5   1
//	/ \ / \
//
// 6  2 0  8
//
//	 / \
//	7   4
