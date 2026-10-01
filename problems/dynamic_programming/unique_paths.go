package dynamic_programming

// LC 62. Unique Paths
//
// There is a robot on an m x n grid. The robot is initially located at the
// top-left corner (i.e., grid[0][0]). The robot tries to move to the
// bottom-right corner (i.e., grid[m - 1][n - 1]). The robot can only move
// either down or right at any point in time.
//
// Given the two integers m and n, return the number of possible unique paths
// that the robot can take to reach the bottom-right corner.
//
// The test cases are generated so that the answer will be less than or equal
// to 2 * 10^9.
//
// Example 1:
//   Input:  m = 3, n = 7
//   Output: 28
//
// Example 2:
//   Input:  m = 3, n = 2
//   Output: 3
//   Explanation: From the top-left corner, there are a total of 3 ways to
//                reach the bottom-right corner:
//     1. Right -> Down -> Down
//     2. Down -> Down -> Right
//     3. Down -> Right -> Down
//
// Constraints:
//   - 1 <= m, n <= 100
//
// Категория: 2D Grid DP.
// Рекуррентность: dp[i][j] = dp[i-1][j] + dp[i][j-1].
// Также решается комбинаторно: C(m+n-2, m-1).

func uniquePaths(m int, n int) int {
	if m < 1 || n < 1 {
		return 0
	}
	if m == 1 && n == 1 {
		return 1
	}
	arrm := make([][]int, m+1)
	for i := 0; i <= m; i++ {
		arrm[i] = make([]int, n+1)
	}
	return uniquePathsHelper(m, n, arrm)
}

func uniquePathsHelper(m int, n int, arr [][]int) int {
	if m < 1 || n < 1 {
		return 0
	}
	if m == 1 && n == 1 {
		return 1
	}
	if arr[m][n] != 0 {
		return arr[m][n]
	}
	arr[m][n] = uniquePathsHelper(m-1, n, arr) + uniquePathsHelper(m, n-1, arr)
	return arr[m][n]
}
