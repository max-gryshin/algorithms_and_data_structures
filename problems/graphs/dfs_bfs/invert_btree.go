package dfs_bfs

import "algorithms_and_data_structures/data_structure"

func invertBtree(root *data_structure.Node) *data_structure.Node {
	return dfsInvertBtree(root)
}

func dfsInvertBtree(node *data_structure.Node) *data_structure.Node {
	if node == nil {
		return nil
	}

	node.Left, node.Right = node.Right, node.Left
	node.Left = dfsInvertBtree(node.Left)
	node.Right = dfsInvertBtree(node.Right)

	return node
}
