package sliding_window

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

func longestSubstringV2(s string) int {
	runes := []rune(s)
	lastSeen := make(map[rune]int)
	left, result := 0, 0
	for right, char := range runes {
		if index, ok := lastSeen[char]; ok && index >= left {
			left = index + 1
		}
		lastSeen[char] = right

		length := right - left + 1
		result = max(result, length)
	}
	return result
}

func longestSubstringV3(s string) string {
	runes := []rune(s)
	lastSeen := make(map[rune]int)

	start := 0
	maxStart := 0
	maxLen := 0

	for current, char := range runes {
		// 1. Сдвигаем левую границу окна за предыдущее вхождение дубликата
		if lastSeenIndex, ok := lastSeen[char]; ok && lastSeenIndex >= start {
			start = lastSeenIndex + 1
		}

		// 2. Обновляем позицию символа
		lastSeen[char] = current

		// 3. Вычисляем длину текущего окна без аллокации среза
		currentLen := current - start + 1

		// 4. Если текущее окно длиннее рекордного — запоминаем его координаты
		if currentLen > maxLen {
			maxLen = currentLen
			maxStart = start
		}
	}

	// Делаем срез только ОДИН раз в самом конце
	return string(runes[maxStart : maxStart+maxLen])
}

//-----------------
// a | b | b | a |
//----------------
