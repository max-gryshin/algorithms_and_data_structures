package sliding_window

// Longest substring without repeating characters

// b | a | b | a |
// L	   R
// b | a | b | b | a |
// L	       R

// longestSubstring - iterates over the string
// for each unique character, increase the counter and store its position
// if character already exists then shift the left pointer
// calculate the longest substring which is window size=right-left+1
// take the bigger substring between the current and existing one.
func longestSubstring(s string) int {
	var res, left, windowSize int
	windowUniqueChars := make(map[rune]int)
	runes := []rune(s)
	for right, char := range runes {
		position, ok := windowUniqueChars[char]
		if ok {
			// if char already exists in the map then shift left pointer
			left = max(left, position+1)
		}
		// increase the counter of character
		// and store its position
		windowUniqueChars[char] = right
		// calculate window size
		windowSize = right - left + 1
		// ensure the window size bigger than longest substring
		res = max(res, windowSize)
	}
	return res
}
