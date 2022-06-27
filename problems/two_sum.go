package main

func twoSum(nums []int, target int) []int {
	numsLen := len(nums)
	if numsLen == 2 {
		return []int{0, 1}
	}
	k := numsLen - 1
	for i := 0; i < k; i++ {
		for j := i + 1; j < numsLen; j++ {
			if (nums[i] + nums[j]) == target {
				return []int{i, j}
			}
			if (nums[k] + nums[i]) == target {
				return []int{k, i}
			}
			if (nums[k] + nums[j]) == target {
				return []int{k, j}
			}
		}
		k--
	}

	return nil
}
