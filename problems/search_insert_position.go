package problems

func searchInsert(nums []int, target int) int {
	var res int
	for i := 0; i < len(nums); i++ {
		if nums[i] == target {
			res = i
			break
		}
		if i > 0 && nums[i-1] < target && i < len(nums) && target < nums[i] {
			res = i
			break
		}
		if i == len(nums)-1 && target > nums[i] {
			res = i + 1
		}
	}
	return res
}
