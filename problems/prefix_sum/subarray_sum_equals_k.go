package prefix_sum

// LC 560. Subarray Sum Equals K
//
// Given an array of integers nums and an integer k, return the total number
// of subarrays whose sum equals to k.
//
// A subarray is a contiguous non-empty sequence of elements within an array.
//
// Example 1:
//   Input:  nums = [1,1,1], k = 2
//   Output: 2
//
// Example 2:
//   Input:  nums = [1,2,3], k = 3
//   Output: 2
//
// Constraints:
//   - 1 <= nums.length <= 2 * 10^4
//   - -1000 <= nums[i] <= 1000
//   - -10^7 <= k <= 10^7

// every point of prefix sum is a boundary between array elements
// [1, 2, 3]
//
//	   1       2       3
//	|---|---|---|---|
//	0   1   2   3   4    ← позиции границ

//              left boundary          right boundary
//                    ↓                       ↓
//nums:        |   2   |   5   |   3   |
//             ↑                           ↑
//             │                           │
//prefix:      0       2       7          10
//                     ↑                   ↑
//                     └────── 8 ──────────┘

func subarraySumK(array []int, k int) int {
	prefixMap := make(map[int]int)
	count := 0
	prefix := 0
	prefixMap[0] = 1
	for _, el := range array {
		// current prefix sum
		prefix += el
		// check if there are sub array with a sum of k
		// k is actually sum of subarray we are looking for
		// [prefix - k] actually means we are looking for left boundary which gives us the sum we are looking for
		if prevPrefix, ok := prefixMap[prefix-k]; ok {
			count += prevPrefix
		}
		// increase prefix count that already been met
		prefixMap[prefix]++
	}
	return count
}
