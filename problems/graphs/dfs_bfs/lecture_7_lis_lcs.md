# Лекция 7: LIS и LCS (Longest Increasing/Common Subsequence)

**Курс**: CS 161 — Design of Algorithms  
**Преподаватель**: Кафедра Computer Science, Stanford University  
**Уровень**: Intermediate

---

## Содержание
1. [Динамическое программирование](#динамическое-программирование)
2. [Longest Increasing Subsequence (LIS)](#longest-increasing-subsequence-lis)
3. [Longest Common Subsequence (LCS)](#longest-common-subsequence-lcs)
4. [Практические примеры](#практические-примеры)
5. [Анализ сложности](#анализ-сложности)

---

## Динамическое программирование

### Принцип DP

Динамическое программирование решает сложные задачи путём:

1. **Разделение** — разложить задачу на подзадачи
2. **Мемоизация** — сохранить результаты подзадач
3. **Комбинирование** — использовать результаты для построения решения

### Свойства DP-задач

```
1. Оптимальная подструктура
   Оптимальное решение содержит оптимальные решения подзадач

2. Перекрывающиеся подзадачи
   Одна подзадача решается многократно

3. Нет циклических зависимостей
   Можно упорядочить вычисления
```

### DP Таблица

```go
// Общий паттерн DP

dp := make([][]int, n+1)
for i := range dp {
	dp[i] = make([]int, m+1)
	for j := range dp[i] {
		dp[i][j] = 0 // базовый случай
	}
}

// Заполнение таблицы
for i := 1; i <= n; i++ {
	for j := 1; j <= m; j++ {
		dp[i][j] = max(dp[i-1][j], dp[i][j-1])
	}
}

return dp[n][m]
```

---

## Longest Increasing Subsequence (LIS)

### Определение

**LIS** — это самая длинная подпоследовательность элементов массива, которая строго возрастает.

```
arr = [10, 9, 2, 5, 3, 7, 101, 18]

LIS: [2, 3, 7, 101] или [2, 3, 7, 18]
Длина: 4
```

### Решение O(n²) — DP

```go
// Найти длину LIS за O(n²)
func LengthOfLIS(nums []int) int {
	n := len(nums)
	if n == 0 {
		return 0
	}
	
	// dp[i] = длина LIS заканчивающейся на nums[i]
	dp := make([]int, n)
	for i := 0; i < n; i++ {
		dp[i] = 1
	}
	
	for i := 1; i < n; i++ {
		for j := 0; j < i; j++ {
			if nums[j] < nums[i] {
				dp[i] = max(dp[i], dp[j]+1)
			}
		}
	}
	
	maxLen := 0
	for _, v := range dp {
		if v > maxLen {
			maxLen = v
		}
	}
	
	return maxLen
}

// Восстановить саму последовательность
func GetLIS(nums []int) []int {
	n := len(nums)
	if n == 0 {
		return []int{}
	}
	
	dp := make([]int, n)
	parent := make([]int, n)
	
	for i := 0; i < n; i++ {
		dp[i] = 1
		parent[i] = -1
	}
	
	for i := 1; i < n; i++ {
		for j := 0; j < i; j++ {
			if nums[j] < nums[i] && dp[j]+1 > dp[i] {
				dp[i] = dp[j] + 1
				parent[i] = j
			}
		}
	}
	
	// Найти позицию максимума
	maxLen := 0
	maxIdx := -1
	for i := 0; i < n; i++ {
		if dp[i] > maxLen {
			maxLen = dp[i]
			maxIdx = i
		}
	}
	
	// Восстановить путь
	result := []int{}
	for idx := maxIdx; idx != -1; idx = parent[idx] {
		result = append([]int{nums[idx]}, result...)
	}
	
	return result
}
```

### Решение O(n log n) — Binary Search

```go
// Оптимальное решение за O(n log n)
func LengthOfLISOptimal(nums []int) int {
	if len(nums) == 0 {
		return 0
	}
	
	// tails[i] = наименьший конец LIS длины i+1
	tails := []int{}
	
	for _, num := range nums {
		// Найти позицию для num используя бинарный поиск
		pos := binarySearchLeftmost(tails, num)
		
		if pos == len(tails) {
			tails = append(tails, num)
		} else {
			tails[pos] = num
		}
	}
	
	return len(tails)
}

// Найти leftmost позицию где можно вставить target
func binarySearchLeftmost(arr []int, target int) int {
	left, right := 0, len(arr)
	
	for left < right {
		mid := (left + right) / 2
		if arr[mid] < target {
			left = mid + 1
		} else {
			right = mid
		}
	}
	
	return left
}
```

### Сравнение подходов

| Подход | Время | Память | Гибкость |
|--------|-------|--------|----------|
| **DP O(n²)** | O(n²) | O(n) | Можно восстановить последовательность |
| **Бинпоиск O(n log n)** | O(n log n) | O(n) | Только длину |

---

## Longest Common Subsequence (LCS)

### Определение

**LCS** — это самая длинная подпоследовательность, которая присутствует в обеих строках (в том же порядке, но не обязательно подряд).

```
text1 = "abcde"
text2 = "ace"

LCS: "ace"
Длина: 3

text1 = "abc"
text2 = "abc"

LCS: "abc"
```

### Решение — 2D DP

```go
// Найти длину LCS за O(m*n)
func LongestCommonSubsequence(text1, text2 string) int {
	m, n := len(text1), len(text2)
	
	// dp[i][j] = длина LCS для text1[0:i] и text2[0:j]
	dp := make([][]int, m+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
	}
	
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if text1[i-1] == text2[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = max(dp[i-1][j], dp[i][j-1])
			}
		}
	}
	
	return dp[m][n]
}

// Восстановить саму LCS
func GetLCS(text1, text2 string) string {
	m, n := len(text1), len(text2)
	
	dp := make([][]int, m+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
	}
	
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if text1[i-1] == text2[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = max(dp[i-1][j], dp[i][j-1])
			}
		}
	}
	
	// Восстановить последовательность
	result := []byte{}
	i, j := m, n
	
	for i > 0 && j > 0 {
		if text1[i-1] == text2[j-1] {
			result = append([]byte{text1[i-1]}, result...)
			i--
			j--
		} else if dp[i-1][j] > dp[i][j-1] {
			i--
		} else {
			j--
		}
	}
	
	return string(result)
}

// Оптимизация памяти до O(n)
func LongestCommonSubsequenceSpaceOptimized(text1, text2 string) int {
	m, n := len(text1), len(text2)
	
	// Использовать только два ряда
	prev := make([]int, n+1)
	curr := make([]int, n+1)
	
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if text1[i-1] == text2[j-1] {
				curr[j] = prev[j-1] + 1
			} else {
				curr[j] = max(prev[j], curr[j-1])
			}
		}
		prev, curr = curr, prev
	}
	
	return prev[n]
}
```

### Визуализация DP таблицы

```
       ""  a  c  e
    "" 0   0  0  0
    a  0   1  1  1
    b  0   1  1  1
    c  0   1  2  2
    d  0   1  2  2
    e  0   1  2  3

LCS("abcde", "ace") = 3
```

---

## Практические примеры

### Пример 1: Edit Distance

```go
// Найти минимальное количество операций редактирования (Levenstein Distance)
func MinDistance(word1, word2 string) int {
	m, n := len(word1), len(word2)
	
	// dp[i][j] = минимальные операции для word1[0:i] -> word2[0:j]
	dp := make([][]int, m+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
	}
	
	// Базовые случаи
	for i := 0; i <= m; i++ {
		dp[i][0] = i // delete
	}
	for j := 0; j <= n; j++ {
		dp[0][j] = j // insert
	}
	
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if word1[i-1] == word2[j-1] {
				dp[i][j] = dp[i-1][j-1]
			} else {
				dp[i][j] = 1 + min(
					dp[i-1][j],   // delete
					min(dp[i][j-1],   // insert
						dp[i-1][j-1]), // replace
				)
			}
		}
	}
	
	return dp[m][n]
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
```

### Пример 2: Distinct Subsequences

```go
// Найти количество различных подпоследовательностей
func NumDistinct(s, t string) int {
	m, n := len(s), len(t)
	
	// dp[i][j] = количество подпоследовательностей t[0:j] в s[0:i]
	dp := make([][]int, m+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
	}
	
	// Базовый случай: пустая строка есть в любой строке один раз
	for i := 0; i <= m; i++ {
		dp[i][0] = 1
	}
	
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			dp[i][j] = dp[i-1][j] // не используем s[i-1]
			
			if s[i-1] == t[j-1] {
				dp[i][j] += dp[i-1][j-1] // используем s[i-1]
			}
		}
	}
	
	return dp[m][n]
}
```

### Пример 3: Interleaving String

```go
// Проверить является ли s3 интерлейвингом s1 и s2
func IsInterleave(s1, s2, s3 string) bool {
	if len(s1)+len(s2) != len(s3) {
		return false
	}
	
	m, n := len(s1), len(s2)
	
	// dp[i][j] = является ли s3[0:i+j] интерлейвингом s1[0:i] и s2[0:j]
	dp := make([][]bool, m+1)
	for i := range dp {
		dp[i] = make([]bool, n+1)
	}
	
	dp[0][0] = true
	
	for i := 1; i <= m; i++ {
		dp[i][0] = dp[i-1][0] && s1[i-1] == s3[i-1]
	}
	
	for j := 1; j <= n; j++ {
		dp[0][j] = dp[0][j-1] && s2[j-1] == s3[j-1]
	}
	
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			idx := i + j - 1
			dp[i][j] = (dp[i-1][j] && s1[i-1] == s3[idx]) ||
				(dp[i][j-1] && s2[j-1] == s3[idx])
		}
	}
	
	return dp[m][n]
}
```

---

## Анализ сложности

### LIS

| Алгоритм | Время | Память | Примечание |
|----------|-------|--------|-----------|
| **DP O(n²)** | O(n²) | O(n) | Простой, восстановление |
| **Бинпоиск O(n log n)** | O(n log n) | O(n) | Быстрый |

### LCS и связанные

| Задача | Время | Память |
|--------|-------|--------|
| **LCS** | O(m·n) | O(m·n) |
| **LCS (оптимизация)** | O(m·n) | O(min(m,n)) |
| **Edit Distance** | O(m·n) | O(m·n) |
| **Distinct Subsequences** | O(m·n) | O(m·n) |

---

## DP таблица для LIS

```
nums = [10, 9, 2, 5, 3, 7, 101, 18]

dp[i] = длина LIS заканчивающейся на nums[i]

i:   0   1   2   3   4   5   6    7
nums: 10  9   2   5   3   7   101  18
dp:   1   1   1   2   2   3   4    4

Ответ: max(dp) = 4
```

---

## DP таблица для LCS

```
text1 = "abcde"
text2 = "ace"

     ""  a   c   e
""    0   0   0   0
a     0   1   1   1
b     0   1   1   1
c     0   1   2   2
d     0   1   2   2
e     0   1   2   3

Ответ: dp[5][3] = 3
LCS: "ace"
```

---

## Рекомендуемые задачи на LeetCode

- [Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/) — классическая LIS
- [Russian Doll Envelopes](https://leetcode.com/problems/russian-doll-envelopes/) — 2D LIS
- [Longest Increasing Subsequence II](https://leetcode.com/problems/longest-increasing-subsequence-ii/) — модификация
- [Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/) — LCS
- [Edit Distance](https://leetcode.com/problems/edit-distance/) — Levenstein
- [Distinct Subsequences](https://leetcode.com/problems/distinct-subsequences/) — подсчёт
- [Interleaving String](https://leetcode.com/problems/interleaving-string/) — проверка

---

## Важные паттерны

### 1D DP (LIS)
```
dp[i] зависит от dp[j] где j < i
```

### 2D DP (LCS)
```
dp[i][j] зависит от dp[i-1][j-1], dp[i-1][j], dp[i][j-1]
```

### Восстановление решения
```
Сохранять parent/previous для восстановления пути
Идти от конца к началу
```

---

## Заключение

LIS и LCS — это классические примеры DP:

1. **LIS** — демонстрирует 1D DP и оптимизацию через бинпоиск
2. **LCS** — демонстрирует 2D DP и восстановление решения
3. **Edit Distance** — показывает применение к строкам
4. **Distinct Subsequences** — считающий вариант DP

Эти задачи часто встречаются в интервью и являются основой для понимания более сложных DP задач.

---

**CS 161 — Design of Algorithms**  
*Stanford University, Department of Computer Science*
