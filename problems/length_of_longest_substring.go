package problems

func lengthOfLongestSubstring(s string) int {
	if s == "" {
		return 0
	}
	chars := make([]string, 0)
	runes := []rune(s)
	chars = append(chars, string(runes[0]))
	runes = append([]rune{}, runes[1:]...)
	lenLongestSubstr := 0
	charsLen := len(chars)
OUTER:
	for _, ch := range runes {
		charsLen = len(chars)
		for k, inCh := range chars {
			if inCh == string(ch) {
				chars = chars[k+1:]
				chars = append(chars, string(ch))
				if charsLen > lenLongestSubstr {
					lenLongestSubstr = charsLen
				}
				continue OUTER
			}
		}
		chars = append(chars, string(ch))
		if charsLen > lenLongestSubstr {
			lenLongestSubstr = charsLen
		}
	}
	if len(chars) > lenLongestSubstr {
		return len(chars)
	}
	return lenLongestSubstr
}
