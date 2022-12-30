package problems

func maxArea(n []int) int {
	end := len(n) - 1
	start := 0
	result := 0
	area := result
	h := 0
	left := 0
	right := 0
	for start < end {
		if h < n[start] {
			left = n[start]
		}
		if h < n[end] {
			right = n[end]
		}
		if left < right {
			h = left
			start++
		} else {
			h = right
			end--
		}
		area = h * (end - start + 1)
		if area > result {
			result = area
		}
	}
	return result
}
