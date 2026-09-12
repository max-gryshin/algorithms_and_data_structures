package two_pointers

// Input:  ['h','e','l','l','o']
// Output: ['o','l','l','e','h']
//
// Input:  ['H','a','n','n','a','h']
// Output: ['h','a','n','n','a','H']
func reverseString(s []rune) []rune {
	left := 0
	right := len(s) - 1
	for left < right {
		s[left], s[right] = s[right], s[left]
		left++
		right--
	}
	return s
}
