package dynamic_programming

// LC 322. Coin Change
//
// You are given an integer array coins representing coins of different
// denominations and an integer amount representing a total amount of money.
//
// Return the fewest number of coins that you need to make up that amount. If
// that amount of money cannot be made up by any combination of the coins,
// return -1.
//
// You may assume that you have an infinite number of each kind of coin.
//
// Example 1:
//   Input:  coins = [1,2,5], amount = 11
//   Output: 3
//   Explanation: 11 = 5 + 5 + 1
//
// Example 2:
//   Input:  coins = [2], amount = 3
//   Output: -1
//
// Example 3:
//   Input:  coins = [1], amount = 0
//   Output: 0
//
// Constraints:
//   - 1 <= coins.length <= 12
//   - 1 <= coins[i] <= 2^31 - 1
//   - 0 <= amount <= 10^4
//
// Категория: unbounded knapsack (монет каждого номинала неограниченно).
// Рекуррентность: dp[a] = min(dp[a - c] + 1) для каждой монеты c <= a.

// 1. state - number of coins - where summ of them equals amount
// 2. transition - iterate over array and multiply coin until it reaches amount
// 3. base - dp[0] = ???
// 4. traversal order -
// 5. result -
func coinChange(coins []int, amount int) int {
	dp := make(map[int]int)
	dp[0] = 0
	for currentAmount := 1; currentAmount <= amount; currentAmount++ {
		for _, coin := range coins {
			if currentAmount < coin {
				continue
			}

			previousAmount := currentAmount - coin
			if count, ok := dp[previousAmount]; ok {
				newCount := count + 1
				if current, ok := dp[currentAmount]; !ok {
					dp[currentAmount] = newCount
				} else {
					dp[currentAmount] = min(current, newCount)
				}
			}
		}
	}

	if result, ok := dp[amount]; ok {
		return result
	}

	return -1
}
