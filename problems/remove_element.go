package problems

import (
	"slices"
)

// original := []int{1, 2, 3, 4, 5}
// result := RemoveFromArray(original, 3) - [1 2 4 5]
// original = [1 2 4 5 5] - since original still have len=5 and the caller see
// that memory has changed where elements shifted left and in the last place a duplicate has appeared
func RemoveFromArray(nums []int, n int) []int {
	for i := 0; i < len(nums); i++ {
		if nums[i] == n {
			return append(nums[:i], nums[i+1:]...)
		}
	}

	return nums
}

func RemoveFromArraySafe(nums []int, n int) []int {
	for i := 0; i < len(nums); i++ {
		if nums[i] == n {
			newArr := make([]int, 0, len(nums)-1)
			newArr = append(newArr, nums[:i]...)
			return append(newArr, nums[i+1:]...)
		}
	}

	return nums
}

func RemoveFromArrayModern(nums []int, n int) []int {
	for i, v := range nums {
		if v == n {
			return slices.Delete(nums, i, i+1)
		}
	}
	return nums
}
