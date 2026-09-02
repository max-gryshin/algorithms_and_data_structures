package prefix_sum

import "testing"

// LC 974. Subarray Sums Divisible by K
//
// Given an integer array nums and an integer k, return the number of
// non-empty subarrays that have a sum divisible by k.
//
// A subarray is a contiguous part of an array.
//
// Example 1:
//   Input:  nums = [4,5,0,-2,-3,1], k = 5
// 					[4,9,9, 7, 4,5]
//					 | |
//					 %5
//					[0,]
//   Output: 7
//   Explanation: There are 7 subarrays with a sum divisible by k = 5:
//     [4, 5, 0, -2, -3, 1], [5], [5, 0], [5, 0, -2, -3], [0], [0, -2, -3],
//     [-2, -3]
//
// Example 2:
//   Input:  nums = [5], k = 9
//   Output: 0
//
// Constraints:
//   - 1 <= nums.length <= 3 * 10^4
//   - -10^4 <= nums[i] <= 10^4
//   - 2 <= k <= 10^4

func TestSubArrSumsDivByK(t *testing.T) {
	table := []struct {
		nums     []int
		k        int
		expected int
	}{
		{
			nums:     []int{4, 5, 0, -2, -3, 1},
			k:        5,
			expected: 7,
		},
	}

	for _, row := range table {
		res := SubArrSumsDivByK(row.nums, row.k)
		if res != row.expected {
			t.Errorf("error expected %d, but got %d", row.expected, res)
		}
	}
}
