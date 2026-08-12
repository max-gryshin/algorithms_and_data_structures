package problems

import (
	"slices"
	"strings"
)

// Vertical Scanning
// сложность по времени будет всего O(N * M),
// так как нам не нужно переупорядочивать весь массив:
func longestCommonPrefixFast(strs []string) string {
	if len(strs) == 0 {
		return ""
	}

	// Берем первое слово за эталон
	first := []rune(strs[0])
	for i := 0; i < len(first); i++ {
		char := first[i]
		// Проверяем этот же символ у всех остальных слов
		for j := 1; j < len(strs); j++ {
			word := []rune(strs[j])
			// Если дошли до конца слова или символы не совпали
			if i >= len(word) || word[i] != char {
				return string(first[:i])
			}
		}
	}
	return string(first)
}

// Horizontal Scanning
func longestCommonPrefix(strs []string) string {
	// 1. Защита от пустого массива
	if len(strs) == 0 {
		return ""
	}

	// Переводим префикс в руны для безопасной работы с UTF-8
	prefix := []rune(strs[0])

	for i := 1; i < len(strs); i++ {
		// 2. Пока текущая строка НЕ начинается с prefix
		for !strings.HasPrefix(strs[i], string(prefix)) {
			// Отрезаем по одной руне с конца
			prefix = prefix[:len(prefix)-1]

			// 3. Защита от выхода за границы: если префикс уменьшился до 0,
			// значит общего префикса нет вообще
			if len(prefix) == 0 {
				return ""
			}
		}
	}

	return string(prefix)
}

func longestCommonPrefixV2(strs []string) string {
	// 1. Защита от пустого массива
	if len(strs) == 0 {
		return ""
	}

	strCopy := make([]string, len(strs))
	// защита от изменения состояния
	copy(strCopy, strs)
	// 2. Сортируем срез
	slices.Sort(strs)

	// 3. Берем первое и последнее слово
	first := []rune(strCopy[0])
	last := []rune(strCopy[len(strCopy)-1])

	minWord := min(len(first), len(last))

	// 4. Ищем индекс первого расхождения
	idx := 0
	for idx < minWord {
		if first[idx] != last[idx] {
			break
		}
		idx++
	}

	// 5. Возвращаем срез без дополнительных append
	return string(first[:idx])
}
