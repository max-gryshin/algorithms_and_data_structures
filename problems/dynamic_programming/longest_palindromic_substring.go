package dynamic_programming

// LC 5. Longest Palindromic Substring
//
// Given a string s, return the longest palindromic substring in s.
//
// Example 1:
//   Input:  s = "babad"
//   Output: "bab"
//   Explanation: "aba" is also a valid answer.
//
// Example 2:
//   Input:  s = "cbbd"
//   Output: "bb"
//
// Constraints:
//   - 1 <= s.length <= 1000
//   - s consist of only digits and English letters.
//
// Категория: substring / palindrome DP.
// Состояние: dp[i][j] = является ли s[i..j] палиндромом.
// Переход: dp[i][j] = (s[i] == s[j]) && (j - i < 2 || dp[i+1][j-1]).
// Обход: по возрастанию длины подстроки. O(n²) время и память.
//
// Альтернатива: expand-around-center — O(n²) время, O(1) память.
// Продвинутое: алгоритм Манакера — O(n).
