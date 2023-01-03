package problems

func plusOne(digits []int) []int {
	length := len(digits) - 1
	var tmp int
	if digits[length] <= 8 {
		digits[length]++
		return digits
	}
	if digits[length] == 9 {
		tmp++
	}
	for i := len(digits) - 1; i >= 0; i-- {
		if tmp > 0 {
			if digits[i] == 9 {
				if i == 0 {
					digits[i] = 1
					digits = append(digits, 0)
					break
				}
				digits[i] = 0
				tmp++
				continue
			}
			tmp = 0
			digits[i]++
		}
	}
	return digits
}
