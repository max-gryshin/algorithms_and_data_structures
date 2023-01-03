package problems

func removeDuplicates(nums []int) int {
	var tmp, res int
	for i := 0; i < len(nums); {
		tmp = nums[i]
		if i > 0 && tmp == nums[i-1] {
			nums = append(nums[:i], nums[i+1:]...)
			continue
		}
		i++
		res++
	}
	return res
}
