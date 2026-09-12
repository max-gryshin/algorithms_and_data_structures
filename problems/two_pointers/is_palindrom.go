package two_pointers

import (
	"strings"
	"unicode"
)

//Input:  "A man, a plan, a canal: Panama"
//Output: true
//
//Input:  "race a car"
//Output: false
//
//Input:  " "
//Output: true

func isPalindrom(s string) bool {
	if s == "" {
		return true
	}
	runes := []rune(strings.ToLower(s))
	left := 0
	right := len(runes) - 1
	for left < right {
		if !unicode.IsLetter(runes[left]) && !unicode.IsDigit(runes[left]) {
			left++
			continue
		}
		if !unicode.IsLetter(runes[right]) && !unicode.IsDigit(runes[right]) {
			right--
			continue
		}
		if runes[left] != runes[right] {
			return false
		}
		left++
		right--
	}
	return true
}
