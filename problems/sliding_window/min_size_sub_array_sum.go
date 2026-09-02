package sliding_window

// MinSizeSubArraySum nums = [2,3,1,2,4,3]
// MinSizeSubArraySum nums = [1,3,1,3,4,3]
// target = 7
//
// Output: 2
func MinSizeSubArraySum(nums []int, target int) int {
	var minSize, left, currentSum int
	for right, _ := range nums {
		// calculate sum of subarray
		currentSum += nums[right]
		// while the current sum >= target
		// 1. calculate minSize
		// 2. shift left pointer
		// 3. decrease the left boundary from subarray
		// to keep the sum of elements only from left to right
		for currentSum >= target {
			if minSize == 0 {
				minSize = right - left + 1
			} else {
				minSize = min(minSize, right-left+1)
			}
			currentSum -= nums[left]
			left++
		}
	}
	return minSize
}
