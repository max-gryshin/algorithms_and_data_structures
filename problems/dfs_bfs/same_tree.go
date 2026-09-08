package dfs_bfs

import "algorithms_and_data_structures/data_structure"

//   1          1
//	/ \        / \
//
// 2   3      2   3
//
//	true

//	  1          1
//	/            \
//
// 2              2
//
//	false
func sameTree(p, q *data_structure.Node) bool {
	return dfsSameTree(p, q)
}

func dfsSameTree(p, q *data_structure.Node) bool {
	if p == nil || q == nil {
		return p == q
	}
	if p.Val != q.Val {
		return false
	}
	return dfsSameTree(p.Left, q.Left) && dfsSameTree(p.Right, q.Right)
}
