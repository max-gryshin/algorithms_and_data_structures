package two_pointers

import "slices"

func threeSum(nums []int) [][]int {
	res := make([][]int, 0)
	slices.Sort(nums)
	for i := 0; i < len(nums)-2; i++ {
		if i > 0 && nums[i] == nums[i-1] {
			continue
		}
		left := i + 1
		right := len(nums) - 1
		for left < right {
			sum := nums[left] + nums[right] + nums[i]
			if sum < 0 {
				left++
				continue
			}
			if sum > 0 {
				right--
				continue
			}

			if sum == 0 {
				newArr := []int{nums[i], nums[left], nums[right]}
				if len(res) == 0 || !slices.Equal(res[len(res)-1], newArr) {
					res = append(res, newArr)
				}
				left++
				right--
			}
		}
	}

	return res
}
