package two_pointers

func removeDuplicates(nums []int) int {
	var k int
	if len(nums) == 0 { // массив пустой -  ничего не делаем
		return 0
	}
	if len(nums) == 1 {
		return 1 // массив 1 элемент просто возрващаем 1 и тоже ничгео не делаем
	}
	// счетчик уникальных элементов
	k = 1
	var uniqPointer int              // первый поинтер
	for i := 1; i < len(nums); i++ { // i - второй поинтер
		if nums[i] == nums[uniqPointer] { // если текущий элемент равен предыдущему то идем дальше
			continue
		} else { // если текущий элемент не равен предыдущему то кладем его в ячейку следующу после uniqPointer
			nums[uniqPointer+1] = nums[i]
			uniqPointer++
			k++
		}
		if uniqPointer == i {
			break // защита на всякий случай
		}
	}

	return k
}

func removeDuplicatesSortedN2(nums []int) int {
	if len(nums) == 0 {
		return 0
	}
	if len(nums) < 3 {
		return len(nums)
	}
	for i := 1; i < len(nums); i++ {
		if i < 2 {
			continue
		}
		if nums[i] == nums[i-2] {
			nums = append(nums[:i-1], nums[i:]...)
			i--
		}
	}

	return len(nums)
}

// 0, 0, 1, 1, 1, 1, 2, 3, 3
// 7
// 0 0 1 1 2 3 3
func removeDuplicatesSorted(nums []int) int {
	if len(nums) <= 2 {
		return len(nums)
	}
	write := 2
	for read := 2; read < len(nums); read++ {
		if nums[write-2] != nums[read] {
			nums[write] = nums[read]
			write++
		}
	}

	return write
}
