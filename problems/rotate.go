package problems

// Rotate cpu O(n), memory 0(n)
func Rotate(nums []int, k int) []int {
	tempK := k
	if k > len(nums) {
		tempK = k % len(nums)
	}
	start := len(nums) - tempK
	res := make([]int, len(nums))
	copy(res, nums)
	for i := 0; i < len(nums); i++ {
		element := nums[(i+start)%(len(nums))]
		res[i] = element
	}
	return res
}

// RotateSwap cpu O(n), memory 0(1)
func RotateSwap(nums []int, k int) {
	n := len(nums)
	k = k % n
	swap(nums, 0, n-1)
	swap(nums, 0, k-1)
	swap(nums, k, n-1)
}

func swap(nums []int, start int, end int) {
	for start < end {
		nums[start], nums[end] = nums[end], nums[start]
		start++
		end--
	}
}
