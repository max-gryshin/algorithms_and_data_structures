package sliding_window

func findAnagrams(s, p string) []int {
	var (
		res              []int
		left, windowSize int
		// number of characters where frequencies are the same
		matches    int
		pFreq      = make(map[rune]int)
		windowFreq = make(map[rune]int)
		runesS     = []rune(s)
		runesP     = []rune(p)
	)
	// prepare map of frequencies of P characters
	for _, charP := range runesP {
		pFreq[charP]++
	}
	// iterate over S characters
	for right, charS := range runesS {
		// increase counter for current char
		windowFreq[charS]++
		// check if character's frequencies in the window and anagram are the same
		if windowFreq[charS] == pFreq[charS] {
			matches++
		}
		// calculate window size
		windowSize = right - left + 1
		// condition to shift left pointer
		if windowSize > len(runesP) {
			ch := runesS[left]
			// check if the frequency of the character from left pointer is the same as in anagram
			if windowFreq[ch] == pFreq[ch] {
				matches--
			}
			// shift left pointer
			left++
			// decrease frequency of the character
			windowFreq[ch]--
			// decrease window size
			windowSize--
		}
		if matches == len(pFreq) {
			res = append(res, left)
		}
	}
	return res
}
