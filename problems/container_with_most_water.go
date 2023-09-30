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

func maxArea2(height []int) int {
	var (
		end          = len(height) - 1
		z            = end
		start        int
		width        int
		minHeight    int
		maxContainer int
		newArea      int
	)
	if height[start] > height[end] {
		minHeight = height[end]
	} else {
		minHeight = height[start]
	}
	width = end - start
	maxContainer = minHeight * width
	for i := 0; i <= end; {
		if (i > 0 && height[i] > height[i-1]) || (z < end && height[z] > height[z+1]) {
			if height[i] > height[z] {
				minHeight = height[z]
			} else {
				minHeight = height[i]
			}
		}
		width = z - i
		newArea = minHeight * width
		if newArea > maxContainer {
			maxContainer = newArea
		}
		if height[i] > height[z] {
			z--
		} else {
			i++
		}
	}

	return maxContainer
}
