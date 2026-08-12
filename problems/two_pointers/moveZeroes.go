package two_pointers

func moveZeroes(nums []int) {
	write := 0
	//nums := []int{0,1,0,3,12}
	// res  - [1,3,12,0,0]
	for read := 0; read < len(nums); read++ {
		if nums[read] != 0 {
			nums[write] = nums[read]
			write++
		}
	}

	for write < len(nums) {
		nums[write] = 0
		write++
	}
}

// move zeros to the end of slice
// pattern - 2 pointers
func moveZeroesV2(nums []int) {
	write := 0 // pointer for write
	//nums := []int{0,1,0,3,12}
	// res  - [1,3,12,0,0]
	for read := range nums {
		if nums[read] != 0 { // if current element not zero
			// we move it to the left side
			// the tricky moment is that if sequence without zero [1,2,3,4]
			// then write and read pointers are the same - and swap does nothing
			nums[write], nums[read] = nums[read], nums[write]
			// then we shift the pointer to the left side,
			write++
		}
	}
}
