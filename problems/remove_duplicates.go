package problems

func removeDuplicates(nums []int) int {
	var tmp, res int
	for i := 0; i < len(nums); {
		tmp = nums[i]
		if i > 0 && tmp == nums[i-1] {
			nums = append(nums[:i], nums[i+1:]...)
			continue
		}
		i++
		res++
	}
	return res
}

func removeDuplicatesV2(nums []int) int {
	var k int
	if len(nums) == 0 { // массив пустой -  ничего не делаем
		return 0
	}
	if len(nums) == 1 {
		return 1 // массив 1 элемент просто возрващаем 1 и тоже ничгео не делаем
	}
	k = 1
	var uniqPointer int              // первый поинтер
	for i := 1; i < len(nums); i++ { // i - второй поинтер
		if nums[i] == nums[uniqPointer] { // если текущий элемент равен предыдущему то идем дальше
			continue
		} else { // если текущий элемент не равен предыдущему то кладем его ячейку следующу после uniqPointer
			nums[uniqPointer+1] = nums[i]
			uniqPointer++ // смещаем первый поинтер
			k++           // увеличиваем счетчик уникальных элементов
		}
		if uniqPointer == i {
			break // защита на всякий случай
		}
	}

	return k
}
