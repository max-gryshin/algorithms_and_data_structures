package problems

// MaxProfit Best Time to Buy and Sell Stock
// Input: prices = [7,1,5,3,6,4]
// Output: 5
// Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
// Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.
func MaxProfit(prices []int) int {
	var max int
	min := prices[0]
	for i := 1; i < len(prices); i++ {
		if min > prices[i] {
			min = prices[i]
		}
		if prices[i]-min > max {
			max = prices[i] - min
		}
	}

	return max
}

// MaxProfit2 Best Time to Buy and Sell Stock II
// Example 1:
// Input: prices = [7,1,5,3,6,4]
// Output: 7
// Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
// Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
// Total profit is 4 + 3 = 7.
// Example 2:
// Input: prices = [1,2,3,4,5]
// Output: 4
// Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
// Total profit is 4.
func MaxProfit2(prices []int) int {
	res := 0
	n := len(prices)
	minTemp := prices[0]
	for i, price := range prices {
		if i > 0 && i == n-1 && price >= prices[i-1] && price-minTemp > 0 {
			res += price - minTemp
		}
		if i > 0 && price >= prices[i-1] {
			continue
		} else {
			if i > 1 {
				res += prices[i-1] - minTemp
			}
			minTemp = price
		}
	}
	return res
}
