package problems

func mergeSort(a []int, b []int) []int {
	length := len(a) + len(b)
	result := make([]int, length)
	var i, j int
	for i < len(a) && j < len(b) {
		if a[i] < b[j] {
			result[i+j] = a[i]
			i++
		} else {
			result[i+j] = b[j]
			j++
		}
	}
	for i < len(a) {
		result[i+j] = a[i]
		i++
	}
	for j < len(b) {
		result[i+j] = b[j]
		j++
	}

	return result
}
