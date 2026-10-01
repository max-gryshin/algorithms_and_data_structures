package dfs_bfs

import "algorithms_and_data_structures/data_structure"

//     tree
//	    1
//	   / \        sub tree
//	  2   3         2'
//	 / \           / \
//	4   5        4'   5'
//
// Output: true

//   tree
//	    1
//	   / \        sub tree
//	  2   3         2'
//	 / \           / \
//	4   5        4'   5'
//       \
//        6
//
// Output: false

func isSubTree(root *data_structure.Node, subTree *data_structure.Node) bool {
	if subTree == nil {
		return true
	}
	if root == nil {
		return false
	}

	return dfsIsSubTree(root, subTree)
}

func dfsIsSubTree(node *data_structure.Node, subTree *data_structure.Node) bool {
	if node == nil && subTree == nil {
		return true
	}
	if node == nil && subTree != nil {
		return false
	}
	// todo: check is it correct
	if subTree == nil && node != nil {
		return false
	}
	if node.Val == subTree.Val {
		if !dfsSameSubTree(node, subTree) {
			return false
		}
	}

	return dfsIsSubTree(node.Left, subTree) || dfsIsSubTree(node.Right, subTree)
}

func dfsSameSubTree(node *data_structure.Node, subTree *data_structure.Node) bool {
	if node == nil && subTree == nil {
		return true
	}
	if node == nil && subTree != nil {
		return false
	}
	if subTree == nil && node != nil {
		return false
	}
	if node.Val != subTree.Val {
		return false
	}
	return dfsSameSubTree(node.Left, subTree.Left) && dfsSameSubTree(node.Right, subTree.Right)
}
