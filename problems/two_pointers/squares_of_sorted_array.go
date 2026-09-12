package two_pointers

// Input:  [-4,-1,0,3,10]
// Output: [0,1,9,16,100]
//
// Input:  [-7,-3,2,3,11]
// Output: [4,9,9,49,121]
//
// Input:  [-100,-90,-20,3,11]
// Output: [4,9,9,49,121]
func squaresOfSortedArray(nums []int) []int {
	res := make([]int, len(nums))
	resIndex := len(nums) - 1
	left := 0
	right := len(nums) - 1
	for left <= right {
		squareLeft := nums[left] * nums[left]
		squareRight := nums[right] * nums[right]
		leftAbs := nums[left]
		rightAbs := nums[right]
		if nums[left] < 0 {
			leftAbs = nums[left] * -1
		}
		if nums[right] < 0 {
			rightAbs = nums[right] * -1
		}
		if leftAbs > rightAbs {
			res[resIndex] = squareLeft
			left++
		} else {
			res[resIndex] = squareRight
			right--
		}
		resIndex--
	}

	return res
}
