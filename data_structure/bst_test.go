package data_structure

import "testing"

func TestNodeAdd(t *testing.T) {
	table := []struct {
		bst           *Bst
		elements      []int
		expectedDepth int
	}{
		{
			bst:           NewBst(),
			elements:      []int{20, 10, 30, 5, 15, 40},
			expectedDepth: 3,
			//          20
			//         /  \
			//       10    30
			//      / \      \
			//     5  15      40
		},
		{
			bst:           NewBst(),
			elements:      []int{20, 10, 30, 5, 15, 40, 3, 17},
			expectedDepth: 4,
			//          20
			//         /  \
			//       10    30
			//      / \      \
			//     5  15      40
			//    /    \
			//   3      17
		},
	}

	for _, row := range table {
		for i := 0; i < len(row.elements); i++ {
			row.bst.Add(row.elements[i])
		}
		if bstDepth := row.bst.MaxDepth(); bstDepth != row.expectedDepth {
			t.Errorf("expected depth: %d but actual %d", row.expectedDepth, bstDepth)
		}
	}
}
