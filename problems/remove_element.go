package problems

func removeElement(nums []int, val int) int {
	var res int
	for i := 0; i < len(nums); {
		if val == nums[i] {
			nums[i] = nums[len(nums)-1]
			nums[len(nums)-1] = 0
			nums = nums[:len(nums)-1]
			continue
		}
		i++
		res++
	}
	return res
}
