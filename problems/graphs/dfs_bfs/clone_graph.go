package dfs_bfs

type Node struct {
	Val       int
	Neighbors []*Node
}

func cloneGraph(root *Node) *Node {
	cloneMap := make(map[*Node]*Node)
	return dfsCloneGraph(root, cloneMap)
}

func dfsCloneGraph(node *Node, cloneMap map[*Node]*Node) *Node {
	if node == nil {
		return nil
	}

	if clone, ok := cloneMap[node]; ok {
		return clone
	}
	clone := &Node{Val: node.Val}
	cloneMap[node] = clone
	for _, neighbor := range node.Neighbors {
		clone.Neighbors = append(clone.Neighbors, dfsCloneGraph(neighbor, cloneMap))
	}
	return clone
}

// 1
// |
// 2   3   4   5
//     |
//     6    7   8
//         |
//         4    10  9

// dfs(1)
//
//	clone 1
//	map[1] = clone1       ← уже здесь!
//	dfs(2)
//	  clone 2
//	  map[2] = clone2
//	  dfs(3)
//	    clone 3
//	    map[3] = clone3
//	    dfs(1)
//	      1 уже есть в map
//	      → возвращаем clone1
