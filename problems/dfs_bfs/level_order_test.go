package dfs_bfs

import (
	"algorithms_and_data_structures/data_structure"
	"slices"
	"testing"
)

func TestLevelOrderBst(t *testing.T) {
	table := []struct {
		bst    func() *data_structure.Bst
		expect [][]int
	}{
		{
			bst: func() *data_structure.Bst {
				binaryTree := data_structure.NewBst()
				//          20
				//         /  \
				//       10    30
				//      / \      \
				//     5  15      40
				for _, el := range []int{20, 10, 30, 5, 15, 40} {
					binaryTree.Add(el)
				}
				return binaryTree
			},
			expect: [][]int{{20}, {10, 30}, {5, 15, 40}},
		},
		{
			bst: func() *data_structure.Bst {
				binaryTree := data_structure.NewBst()
				//          20
				//         /  \
				//       10    30
				//      / \      \
				//     5  15      40
				//    /    \
				//   3      17
				for _, el := range []int{20, 10, 30, 5, 15, 40, 3, 17} {
					binaryTree.Add(el)
				}
				return binaryTree
			},
			expect: [][]int{{20}, {10, 30}, {5, 15, 40}, {3, 17}},
		},
	}

	for _, row := range table {
		res := levelOrderBst(row.bst())
		if len(res) != len(row.expect) {
			t.Errorf("expected result len %d, but got %d", len(row.expect), len(res))
		}
		for index, levelExpected := range row.expect {
			if !slices.Equal(levelExpected, res[index]) {
				t.Errorf("expected level array to be %v but got %v", levelExpected, res[index])
			}
		}
	}
}
