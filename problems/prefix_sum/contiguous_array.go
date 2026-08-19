package prefix_sum

// Given a binary array nums, return the maximum length of a contiguous subarray with an equal number of 0 and 1.
//Example 1:
//Input: nums = [0,1]
//Output: 2
//Explanation: [0, 1] is the longest contiguous subarray with an equal number of 0 and 1.
//Example 2:
//
//Input: nums = [0,1,0]
//Output: 2
//Explanation: [0, 1] (or [1, 0]) is a longest contiguous subarray with equal number of 0 and 1.
//Example 3:
//
//Input: nums = [0,1,1,1,1,1,0,0,0]
//Output: 6
//Explanation: [1,1,1,0,0,0] is the longest contiguous subarray with equal number of 0 and 1.

func findMaxLength(nums []int) int {
	// balance tracks the balance between 0 and 1
	var balance, maxLength int
	m := make(map[int]int)
	m[0] = -1
	for i := 0; i < len(nums); i++ {
		if nums[i] == 0 {
			balance--
		} else {
			balance++
		}
		// if map already has such a balance, it means
		// that we have subarray with 0 balances
		if prevIndex, ok := m[balance]; ok {
			length := i - prevIndex
			if length > maxLength {
				maxLength = length
			}
		} else {
			// put only if map do not have such balance yet
			// since we need the first index only to take the max length
			m[balance] = i
		}
	}
	return maxLength
}
