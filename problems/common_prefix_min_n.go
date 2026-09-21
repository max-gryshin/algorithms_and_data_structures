package problems

import "slices"

func commonPrefix(words []string, minNumber int) string {
	if len(words) == 0 || minNumber <= 0 || minNumber > len(words) {
		return ""
	}

	slices.Sort(words)

	wordsRunes := make([][]rune, 0, len(words))
	for _, word := range words {
		wordsRunes = append(wordsRunes, []rune(word))
	}

	var longestPrefix string

	for i := 0; i <= len(wordsRunes)-minNumber; i++ {
		firstWord := wordsRunes[i]
		lastWord := wordsRunes[i+minNumber-1]

		limit := min(len(firstWord), len(lastWord))
		longestPrefixLocal := 0

		for k := 0; k < limit; k++ {
			if firstWord[k] != lastWord[k] {
				break
			}
			longestPrefixLocal++
		}

		candidate := string(firstWord[:longestPrefixLocal])

		if len(candidate) > len(longestPrefix) ||
			(len(candidate) == len(longestPrefix) && candidate < longestPrefix) {
			longestPrefix = candidate
		}
	}

	return longestPrefix
}
