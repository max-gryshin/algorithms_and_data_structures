package problems

import (
	"sort"
)

// nums = [1, 11, 111, 1111, 11111]
// n = 22
// result = [1, 11, 22, 111, 1111, 11111]
func AddToSortedArray(nums []int, n int) []int {
	i := sort.SearchInts(nums, n)
	nums = append(nums, 0)
	copy(nums[i+1:], nums[i:])
	nums[i] = n

	return nums
}
