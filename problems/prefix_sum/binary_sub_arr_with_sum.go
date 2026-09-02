package prefix_sum

// nums = [1,0,1,0,1]
// goal = 2
//
// answer = 4
// Explanation: The 4 subarrays are bolded and underlined below:
// [1,0,1]
// [1,0,1,0]
// [0,1,0,1]
// [1,0,1]
func numSubarraysWithSum(nums []int, goal int) int {
	var res, prefix int
	m := make(map[int]int)
	m[0] = 1
	for _, num := range nums {
		prefix += num
		res += m[prefix-goal]
		m[prefix]++
	}
	return res
}
