package prefix_sum

/*
PROBLEM: Zero-Balance Transaction Block

You are given a slice of integers representing financial transactions over the course of a day.
Positive integers represent deposits, and negative integers represent withdrawals.

Your task is to write a function that returns `true` if there is ANY consecutive sequence
of transactions (a block of 1 or more consecutive transactions) that perfectly nets out to exactly zero.
Otherwise, return `false`.

EXAMPLES:
1. Input: []int{4, 2, -3, 1, 6}
   Output: true (because the consecutive block [2, -3, 1] sums to 0)

2. Input: []int{5, -5, 2}
   Output: true (because the consecutive block [5, -5] sums to 0)

3. Input: []int{1, 2, 3, 4, 5}
   Output: false (no consecutive block sums to 0)

4. Input: []int{3, 5, -2, -3, 8}
   Output: true (because the consecutive block [5, -2, -3] sums to 0)
*/

// 4, 2, -3, 1, 6
// 4, 6, 3, 4, 10 - balance on each step
// |	    | equal balance on these blocks says there is a zero-sum block

func hasZeroSumBlock(transactions []int) bool {
	if len(transactions) == 0 {
		return false
	}
	if len(transactions) == 1 {
		return transactions[0] == 0
	}
	m := make(map[int]bool)
	var prefix int
	for _, transaction := range transactions {
		prefix += transaction
		if prefix == 0 {
			return true
		}
		if _, ok := m[prefix]; ok {
			return true
		}
		m[prefix] = true
	}
	return false
}
