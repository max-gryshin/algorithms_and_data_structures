package dfs_bfs

import "algorithms_and_data_structures/data_structure"

//	    1
//	   / \
//	  2   3
//	 / \
//	4   5
//
// diameter: 3

func diameterBtree(root *data_structure.Node) int {
	var diameter int
	if root == nil {
		return diameter
	}

	dfsDiameterBtree(root, &diameter)

	return diameter
}

func dfsDiameterBtree(node *data_structure.Node, diameter *int) int {
	if node == nil {
		return 0
	}

	left := dfsDiameterBtree(node.Left, diameter)
	right := dfsDiameterBtree(node.Right, diameter)
	*diameter = max(*diameter, left+right)

	return max(left, right) + 1
}
