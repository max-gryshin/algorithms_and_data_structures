package problems

import "algorithms_and_data_structures/algorythm/sort"

func threeSum(nums []int) [][]int {
	var res [][]int
	sort.HeapSort(&nums)
	for i, v := range nums {
		if i > 0 && v == nums[i-1] {
			continue
		}
		l, r := i+1, len(nums)-1
		for l < r {
			if v+nums[l]+nums[r] > 0 {
				r--
			}
			if v+nums[l]+nums[r] < 0 {
				l++
			}
			if r == l {
				break
			}
			if v+nums[l]+nums[r] == 0 {
				res = append(res, []int{v, nums[l], nums[r]})
				l++
				for l < r && nums[l] == nums[l-1] {
					l++
				}
			}
		}
	}
	return res
}
