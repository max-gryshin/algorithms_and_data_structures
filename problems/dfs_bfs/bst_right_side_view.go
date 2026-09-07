package dfs_bfs

import "algorithms_and_data_structures/data_structure"

// Binary Tree Right Side View

func rightSideView(root *data_structure.Node) []int {
	if root == nil {
		return nil
	}
	res := make([]int, 0)
	queue := []*data_structure.Node{root}

	//          20
	//         /  \
	//       10    30
	//      / \      \
	//     5  15      40
	// res: [20, 30, 40]

	//       1
	//     / \
	//    2   3
	//   /
	//  4
	// res: [1,3,4]
	for len(queue) > 0 {
		currentQueueLen := len(queue)
		var lastEl *data_structure.Node
		for i := 0; i < currentQueueLen; i++ {
			lastEl = queue[0]
			queue = queue[1:]
			if lastEl.Right != nil {
				queue = append(queue, lastEl.Right)
			}
			if lastEl.Left != nil {
				queue = append(queue, lastEl.Left)
			}
		}
		if lastEl != nil {
			res = append(res, lastEl.Val)
		}
	}

	return res
}
