package hashmap

func twoSum(nums []int, target int) []int {
	m := make(map[int]int)

	for index, num := range nums {
		completes := target - num
		if idx, ok := m[completes]; ok {
			return []int{index, idx}
		}
		m[num] = index
	}

	return nil
}
