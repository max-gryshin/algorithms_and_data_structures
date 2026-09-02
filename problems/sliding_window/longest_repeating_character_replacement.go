package sliding_window

// LongestRepeatingCharacterReplacement
// right++
// count[s[right]]++
// maxFrequency = max(maxFrequency, count[s[right]])
//
// while windowSize - maxFrequency > k:
//
//	count[s[left]]--
//	left++
//
// answer = max(answer, windowSize)
func LongestRepeatingCharacterReplacement(s string, k int) int {
	var res, windowSize, maxFrequency, left int
	charCounter := make(map[rune]int)
	runes := []rune(s)

	for right, char := range runes {
		// count a repeating character
		charCounter[char]++
		// calculate max frequency
		maxFrequency = max(maxFrequency, charCounter[char])
		// calculate windowSize
		windowSize = right - left + 1
		// if diff between window size and max frequency > k
		// then shift left pointer and recalculate window size
		for windowSize-maxFrequency > k {
			left++
			// store counts of characters which are in the window
			charCounter[runes[left]]--
			windowSize = right - left + 1
		}
		res = max(res, windowSize)
	}

	return res
}
