package two_pointers

// Input:  numbers = [2,7,11,15], target = 9 or 18
// Output: [1,2]
//
// Input:  numbers = [2,3,4], target = 6 or 7
// Output: [1,3] or [2,3]
//
// Input:  numbers = [-1,0], target = -1
// Output: [1,2]
func twoSum(nums []int, target int) []int {
	res := make([]int, 0)
	left := 0
	right := len(nums) - 1
	for left < right {
		sum := nums[left] + nums[right]
		if sum == target {
			res = append(res, left, right)
			break
		}
		if sum < target {
			left++
		}
		if sum > target {
			right--
		}
	}

	return res
}
