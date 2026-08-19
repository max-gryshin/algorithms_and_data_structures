package prefix_sum

// 304. Range Sum Query 2D - Immutable
//Medium
//Topics
//premium lock icon
//Companies
//Given a 2D matrix matrix, handle multiple queries of the following type:
//
//Calculate the sum of the elements of matrix inside the rectangle defined by its upper left corner (row1, col1) and lower right corner (row2, col2).
//Implement the NumMatrix class:
//
//NumMatrix(int[][] matrix) Initializes the object with the integer matrix matrix.
//int sumRegion(int row1, int col1, int row2, int col2) Returns the sum of the elements of matrix inside the rectangle defined by its upper left corner (row1, col1) and lower right corner (row2, col2).
//You must design an algorithm where sumRegion works on O(1) time complexity.

// ┌────────────────────────────┐
// │         :              :   │
// │ overlap :    top       :   │
// │         :              :   │
// │.........┌──────────────┐   │
// │         │              │   │
// │   left  │    target    │   │
// │         │              │   │
// │         └──────────────┘   │
// │                            │
// │              	            │
// └────────────────────────────┘
// target = maxArea - top - left + overlap

type NumMatrix struct {
	prefixMatrix [][]int
}

func ConstructorNumMatrix(matrix [][]int) NumMatrix {
	newM := make([][]int, len(matrix)+1)
	for i, _ := range newM {
		newM[i] = make([]int, len(matrix)+1)
	}

	for i := 1; i < len(matrix)+1; i++ {
		prefix := 0
		for j := 1; j < len(matrix)+1; j++ {
			top := newM[i-1][j]
			topLeftOverlap := newM[i-1][j-1]
			prefix += matrix[i-1][j-1] + top - topLeftOverlap
			newM[i][j] = prefix
		}
	}
	return NumMatrix{prefixMatrix: newM}
}

// SumRegion Optimized
func (nm *NumMatrix) SumRegion(row1 int, col1 int, row2 int, col2 int) int {
	whole := nm.prefixMatrix[row2+1][col2+1]
	top := nm.prefixMatrix[row1][col2+1]
	left := nm.prefixMatrix[row2+1][col1]
	overlap := nm.prefixMatrix[row1][col1]

	return whole - top - left + overlap
}
