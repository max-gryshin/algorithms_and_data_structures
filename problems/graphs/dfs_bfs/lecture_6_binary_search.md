# Лекция 6: Бинарный поиск и Two Pointers

**Курс**: CS 161 — Design of Algorithms  
**Преподаватель**: Кафедра Computer Science, Stanford University  
**Уровень**: Intermediate

---

## Содержание
1. [Классический бинарный поиск](#классический-бинарный-поиск)
2. [Варианты бинарного поиска](#варианты-бинарного-поиска)
3. [Поиск ответа (Answer Search)](#поиск-ответа-answer-search)
4. [Two Pointers паттерн](#two-pointers-паттерн)
5. [Практические примеры](#практические-примеры)
6. [Анализ сложности](#анализ-сложности)

---

## Классический бинарный поиск

### Идея

**Бинарный поиск** работает на **отсортированном массиве**, исключая половину элементов на каждом шаге.

```
arr = [1, 3, 5, 7, 9, 11, 13, 15]
target = 7

Шаг 1: left=0, right=7, mid=3 → arr[3]=7 → FOUND!

Если бы искали 6:
Шаг 1: mid=3 → arr[3]=7 > 6 → right=2
Шаг 2: mid=1 → arr[1]=3 < 6 → left=2
Шаг 3: left > right → NOT FOUND
```

### Реализация

```go
// Классический бинарный поиск
func BinarySearch(arr []int, target int) int {
	left, right := 0, len(arr)-1
	
	for left <= right {
		mid := left + (right-left)/2 // Избегаем overflow
		
		if arr[mid] == target {
			return mid
		} else if arr[mid] < target {
			left = mid + 1 // target в правой половине
		} else {
			right = mid - 1 // target в левой половине
		}
	}
	
	return -1 // Не найдено
}
```

### Граничные случаи

```
Пустой массив       → -1
Один элемент        → проверка
target < arr[0]     → -1
target > arr[n-1]   → -1
Дубликаты           → любой индекс с target
```

---

## Варианты бинарного поиска

### 1. Найти первый индекс (Leftmost)

```go
// Найти первый индекс элемента, >= target
func BinarySearchLeftmost(arr []int, target int) int {
	left, right := 0, len(arr)
	
	for left < right {
		mid := left + (right-left)/2
		if arr[mid] < target {
			left = mid + 1
		} else {
			right = mid
		}
	}
	
	return left
}

// Пример:
// arr = [1, 3, 3, 3, 5], target = 3
// Возвращает 1 (первый индекс с 3)
```

### 2. Найти последний индекс (Rightmost)

```go
// Найти последний индекс элемента, <= target
func BinarySearchRightmost(arr []int, target int) int {
	left, right := 0, len(arr)
	
	for left < right {
		mid := left + (right-left)/2
		if arr[mid] <= target {
			left = mid + 1
		} else {
			right = mid
		}
	}
	
	return left - 1
}

// Пример:
// arr = [1, 3, 3, 3, 5], target = 3
// Возвращает 3 (последний индекс с 3)
```

### 3. Найти диапазон [first, last]

```go
// Найти диапазон индексов с target
func SearchRange(arr []int, target int) []int {
	if len(arr) == 0 {
		return []int{-1, -1}
	}
	
	first := BinarySearchLeftmost(arr, target)
	if first == len(arr) || arr[first] != target {
		return []int{-1, -1}
	}
	
	last := BinarySearchRightmost(arr, target)
	
	return []int{first, last}
}
```

---

## Поиск ответа (Answer Search)

### Концепция

**Answer Search** — используем бинарный поиск на пространстве ответов, а не на массиве.

```
Вместо поиска значения в массиве,
ищем максимальное/минимальное значение, которое удовлетворяет условию.

Функция проверки: для каждого кандидата x, проверяем выполняется ли условие
```

### Шаблон

```
Условие1: если x работает → все y > x работают (монотонное возрастание)
Условие2: если x не работает → все y < x не работают

Ищем граничное значение: максимум x который не работает, или минимум x который работает
```

### Пример 1: Минимальная скорость

```go
// Найти минимальную скорость, чтобы финишить за H часов
func MinimumSpeed(dist []int, hour int) int {
	left, right := 1, 10_000_000
	
	// Проверка: можем ли мы финишить на скорости speed?
	canFinish := func(speed int) bool {
		totalTime := 0
		for _, d := range dist {
			totalTime += (d + speed - 1) / speed // Ceiling division
		}
		return totalTime <= hour
	}
	
	// Ищем минимальное speed где canFinish = true
	for left < right {
		mid := left + (right-left)/2
		if canFinish(mid) {
			right = mid // середовище работает, ищем меньше
		} else {
			left = mid + 1 // не работает, ищем больше
		}
	}
	
	return left
}
```

### Пример 2: Максимальный конфет в сумках

```go
// Максимальное количество конфет, которое мы можем положить в k сумок
func MaximumCandies(candies []int, k int64) int64 {
	maxCandy := int64(0)
	for _, c := range candies {
		if int64(c) > maxCandy {
			maxCandy = int64(c)
		}
	}
	
	left, right := int64(0), maxCandy
	
	canDistribute := func(x int64) bool {
		bags := int64(0)
		for _, c := range candies {
			bags += int64(c) / x
		}
		return bags >= k
	}
	
	for left < right {
		mid := left + (right-left+1)/2
		if canDistribute(mid) {
			left = mid
		} else {
			right = mid - 1
		}
	}
	
	return left
}
```

---

## Two Pointers паттерн

### Идея

**Two Pointers** — используем два указателя на массиве, которые движутся стратегически.

```
Два указателя:
1. Противоположные стороны (встречаются)
2. Один быстро, один медленно
3. Оба движутся в одну сторону
```

### 1. Встречающиеся указатели

```go
// Two Sum в отсортированном массиве
func TwoSum(arr []int, target int) []int {
	left, right := 0, len(arr)-1
	
	for left < right {
		sum := arr[left] + arr[right]
		if sum == target {
			return []int{left, right}
		} else if sum < target {
			left++ // нужно больше, идём вправо
		} else {
			right-- // нужно меньше, идём влево
		}
	}
	
	return []int{-1, -1}
}
```

### 2. Быстрый и медленный указатель

```go
// Удалить дубликаты (in-place)
func RemoveDuplicates(nums []int) int {
	if len(nums) == 0 {
		return 0
	}
	
	slow := 0
	
	for fast := 1; fast < len(nums); fast++ {
		if nums[fast] != nums[slow] {
			slow++
			nums[slow] = nums[fast]
		}
	}
	
	return slow + 1
}

// [1,1,2] → [1,2,_] возвращает 2
```

### 3. Sliding Window (вариант Two Pointers)

```go
// Longest substring без повторяющихся символов
func LengthOfLongestSubstring(s string) int {
	charIndex := make(map[rune]int)
	maxLen := 0
	left := 0
	
	for right := 0; right < len(s); right++ {
		char := rune(s[right])
		
		// Если символ уже в окне, сместить левый указатель
		if idx, ok := charIndex[char]; ok && idx >= left {
			left = idx + 1
		}
		
		charIndex[char] = right
		maxLen = max(maxLen, right-left+1)
	}
	
	return maxLen
}
```

### 4. Container With Most Water

```go
// Найти контейнер с максимальным объёмом
func MaxArea(height []int) int {
	left, right := 0, len(height)-1
	maxArea := 0
	
	for left < right {
		h := min(height[left], height[right])
		w := right - left
		area := h * w
		
		maxArea = max(maxArea, area)
		
		// Двигаем указатель на более короткой линии
		if height[left] < height[right] {
			left++
		} else {
			right--
		}
	}
	
	return maxArea
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```

---

## Практические примеры

### Пример 1: Search in Rotated Sorted Array

```go
// Поиск в развёрнутом отсортированном массиве
func Search(nums []int, target int) int {
	left, right := 0, len(nums)-1
	
	for left <= right {
		mid := left + (right-left)/2
		
		if nums[mid] == target {
			return mid
		}
		
		// Определить какая половина отсортирована
		if nums[left] <= nums[mid] {
			// Левая половина отсортирована
			if nums[left] <= target && target < nums[mid] {
				right = mid - 1
			} else {
				left = mid + 1
			}
		} else {
			// Правая половина отсортирована
			if nums[mid] < target && target <= nums[right] {
				left = mid + 1
			} else {
				right = mid - 1
			}
		}
	}
	
	return -1
}
```

### Пример 2: 3Sum

```go
// Найти все тройки с суммой = 0
func ThreeSum(nums []int) [][]int {
	sort.Ints(nums)
	result := [][]int{}
	
	for i := 0; i < len(nums)-2; i++ {
		// Пропустить дубликаты
		if i > 0 && nums[i] == nums[i-1] {
			continue
		}
		
		// Если все оставшиеся положительны, невозможно
		if nums[i] > 0 {
			break
		}
		
		left, right := i+1, len(nums)-1
		
		for left < right {
			sum := nums[i] + nums[left] + nums[right]
			
			if sum == 0 {
				result = append(result, []int{nums[i], nums[left], nums[right]})
				
				// Пропустить дубликаты
				for left < right && nums[left] == nums[left+1] {
					left++
				}
				for left < right && nums[right] == nums[right-1] {
					right--
				}
				
				left++
				right--
			} else if sum < 0 {
				left++
			} else {
				right--
			}
		}
	}
	
	return result
}
```

---

## Анализ сложности

| Алгоритм | Время | Память |
|----------|-------|--------|
| **Классический BS** | O(log n) | O(1) |
| **Answer Search** | O(log n * check) | O(1) |
| **Two Pointers** | O(n) | O(1) |
| **Sliding Window** | O(n) | O(k) set |

---

## Рекомендуемые задачи на LeetCode

- [Binary Search](https://leetcode.com/problems/binary-search/)
- [Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/)
- [Find First and Last Position](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)
- [Minimum Speed to Finish All Dishes](https://leetcode.com/problems/minimum-speed-to-finish-all-dishes/)
- [Container With Most Water](https://leetcode.com/problems/container-with-most-water/)
- [Remove Duplicates from Sorted Array](https://leetcode.com/problems/remove-duplicates-from-sorted-array/)
- [3Sum](https://leetcode.com/problems/3sum/)
- [Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)

---

## Заключение

1. **Бинарный поиск** — O(log n), требует отсортированность
2. **Answer Search** — бинпоиск на ответах, не значениях
3. **Two Pointers** — O(n), стратегическое движение
4. **Combination** — часто используются вместе

---

**CS 161 — Design of Algorithms**  
*Stanford University, Department of Computer Science*
