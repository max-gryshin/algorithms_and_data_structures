package prefix_sum

import "testing"

func TestConstructorNumMatrix(t *testing.T) {
	table := []struct {
		matrix   [][]int
		expected [][]int
	}{
		{
			matrix: [][]int{
				{1, 2, 3},
				{4, 5, 6},
				{7, 8, 9},
			},
			expected: [][]int{
				{0, 0, 0, 0},
				{0, 1, 3, 6},
				{0, 5, 12, 21},
				{0, 12, 27, 45},
			},
		},
	}

	for _, testCase := range table {
		res := ConstructorNumMatrix(testCase.matrix)
		if !IsMatrixesEqual(res.prefixMatrix, testCase.expected) {
			t.Errorf("Expected %v, got %v", testCase.matrix, res)
		}
	}
}

func TestSumRegion(t *testing.T) {
	table := []struct {
		matrix   [][]int
		row1     int
		col1     int
		row2     int
		col2     int
		expected int
	}{
		{
			matrix: [][]int{
				{1, 2, 3},
				{4, 5, 6},
				{7, 8, 9},
			},
			row1:     0,
			col1:     1,
			row2:     1,
			col2:     1,
			expected: 7,
		},
		{
			matrix: [][]int{
				{1, 2, 3},
				{4, 5, 6},
				{7, 8, 9},
			},
			row1:     0,
			col1:     0,
			row2:     0,
			col2:     0,
			expected: 1,
		},
		{
			matrix: [][]int{
				{1, 2, 3},
				{4, 5, 6},
				{7, 8, 9},
			},
			row1:     0,
			col1:     0,
			row2:     2,
			col2:     2,
			expected: 45,
		},
		{
			matrix: [][]int{
				{1, 2, 3},
				{4, 5, 6},
				{7, 8, 9},
			},
			row1:     1,
			col1:     1,
			row2:     2,
			col2:     2,
			expected: 28,
		},
		{
			matrix: [][]int{
				{1, 2, 3},
				{4, 5, 6},
				{7, 8, 9},
			},
			row1:     2,
			col1:     0,
			row2:     2,
			col2:     2,
			expected: 24,
		},
	}

	for _, testCase := range table {
		numMatrix := ConstructorNumMatrix(testCase.matrix)
		if got := numMatrix.SumRegion(
			testCase.row1,
			testCase.col1,
			testCase.row2,
			testCase.col2,
		); got != testCase.expected {
			t.Errorf("Expected %v, got %v", testCase.expected, got)
		}
	}
}

func IsMatrixesEqual(m1 [][]int, m2 [][]int) bool {
	if len(m1) != len(m2) {
		return false
	}
	for i, row := range m1 {
		for j, _ := range row {
			if m1[i][j] != m2[i][j] {
				return false
			}
		}
	}

	return true
}
