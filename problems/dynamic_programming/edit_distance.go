package dynamic_programming

// LC 72. Edit Distance
//
// Given two strings word1 and word2, return the minimum number of operations
// required to convert word1 to word2.
//
// You have the following three operations permitted on a word:
//   - Insert a character
//   - Delete a character
//   - Replace a character
//
// Example 1:
//   Input:  word1 = "horse", word2 = "ros"
//   Output: 3
//   Explanation:
//     horse -> rorse (replace 'h' with 'r')
//     rorse -> rose  (remove 'r')
//     rose  -> ros   (remove 'e')
//
// Example 2:
//   Input:  word1 = "intention", word2 = "execution"
//   Output: 5
//   Explanation:
//     intention -> inention (remove 't')
//     inention  -> enention (replace 'i' with 'e')
//     enention  -> exention (replace 'n' with 'x')
//     exention  -> exection (replace 'n' with 'c')
//     exection  -> execution (insert 'u')
//
// Constraints:
//   - 0 <= word1.length, word2.length <= 500
//   - word1 and word2 consist of lowercase English letters.
//
// Категория: Levenshtein distance, 2D string DP.
// Рекуррентность:
//   если word1[i-1] == word2[j-1]: dp[i][j] = dp[i-1][j-1]
//   иначе dp[i][j] = 1 + min(dp[i-1][j],       // delete
//                            dp[i][j-1],       // insert
//                            dp[i-1][j-1])     // replace
// База: dp[0][j] = j, dp[i][0] = i.
