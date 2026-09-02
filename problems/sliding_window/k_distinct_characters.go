package sliding_window

// Longest Substring with At Most K Distinct Characters

// s = "eceba", k = 2, res = 3 -> "ece"
func kDistinctCharacters(s string, k int) int {
	var res, left int
	// store frequency of the elements
	m := make(map[rune]int)
	runes := []rune(s)
	for right, char := range runes {
		// increase frequency of the character
		m[char]++
		// len(m) - shows the amount of distinct characters in range [L...R]
		for len(m) > k {
			ch := runes[left]
			// before shifting left pointer remove frequency
			m[ch]--
			// if frequency is 0 remove character at all from the map
			if m[ch] == 0 {
				delete(m, ch)
			}
			left++
		}
		// here we already sure that after iterating len(m) > k the window [L...R]
		// contains no more than K distinct characters so we can just add it to the result
		res = max(res, right-left+1)
	}
	return res
}
