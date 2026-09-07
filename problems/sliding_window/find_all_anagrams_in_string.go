package sliding_window

// Example 1:
//
// Input: s = "cbaebabacd", p = "abc"
// Output: [0,6]
// Explanation:
// The substring with start index = 0 is "cba", which is an anagram of "abc".
// The substring with start index = 6 is "bac", which is an anagram of "abc".
// Example 2:
//
// Input: s = "abab", p = "ab"
// Output: [0,1,2]
// Explanation:
// The substring with start index = 0 is "ab", which is an anagram of "ab".
// The substring with start index = 1 is "ba", which is an anagram of "ab".
// The substring with start index = 2 is "ab", which is an anagram of "ab".
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
		// kep window size no bigger than len of anagram
		// condition to shift left pointer
		if windowSize > len(runesP) {
			leftCharacter := runesS[left]
			// check if the frequency of the character from left pointer is the same as in anagram
			if windowFreq[leftCharacter] == pFreq[leftCharacter] {
				matches--
			}
			// shift left pointer
			left++
			// decrease frequency of the character
			windowFreq[leftCharacter]--
			// decrease window size
			windowSize--
		}
		if matches == len(pFreq) {
			res = append(res, left)
		}
	}
	return res
}
