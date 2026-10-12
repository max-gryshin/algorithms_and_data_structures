# Лекция 13: Bit Manipulation и Bit DP

**Курс**: CS 161 — Design of Algorithms  
**Преподаватель**: Кафедра Computer Science, Stanford University  
**Уровень**: Intermediate

---

## Содержание
1. [Битовые операции](#битовые-операции)
2. [XOR свойства](#xor-свойства)
3. [Bit Manipulation трюки](#bit-manipulation-трюки)
4. [Bitmask DP](#bitmask-dp)
5. [Практические примеры](#практические-примеры)
6. [Анализ сложности](#анализ-сложности)

---

## Битовые операции

### Основные операции

```
AND (&):    1 & 1 = 1,  1 & 0 = 0
OR (|):     1 | 0 = 1,  0 | 0 = 0
XOR (^):    1 ^ 1 = 0,  1 ^ 0 = 1
NOT (~):    ~1 = 0,     ~0 = 1
Сдвиг <<:   x << 1 = x * 2
Сдвиг >>:   x >> 1 = x / 2
```

### Реализация на Go

```go
// AND - пересечение битов
func BitwiseAnd(a, b int) int {
	return a & b
}

// OR - объединение битов
func BitwiseOr(a, b int) int {
	return a | b
}

// XOR - различие битов
func BitwiseXor(a, b int) int {
	return a ^ b
}

// NOT - инверсия (зависит от размера числа)
func BitwiseNot(a int) int {
	return ^a
}

// Левый сдвиг (умножение на 2^k)
func LeftShift(a, k int) int {
	return a << k
}

// Правый сдвиг (деление на 2^k)
func RightShift(a, k int) int {
	return a >> k
}

// Проверить установлен ли бит на позиции k
func IsBitSet(x, k int) bool {
	return (x & (1 << k)) != 0
}

// Установить бит на позиции k
func SetBit(x, k int) int {
	return x | (1 << k)
}

// Очистить бит на позиции k
func ClearBit(x, k int) int {
	return x & ^(1 << k)
}

// Переключить бит на позиции k
func ToggleBit(x, k int) int {
	return x ^ (1 << k)
}

// Получить наименьший установленный бит
func GetLowestSetBit(x int) int {
	return x & (-x)
}

// Удалить наименьший установленный бит
func RemoveLowestSetBit(x int) int {
	return x & (x - 1)
}

// Подсчитать количество установленных битов
func CountBits(x int) int {
	count := 0
	for x > 0 {
		count += x & 1
		x >>= 1
	}
	return count
}

// Проверить является ли число степенью двойки
func IsPowerOfTwo(x int) bool {
	return x > 0 && (x&(x-1)) == 0
}
```

---

## XOR свойства

### Важные свойства

```
1. a ^ a = 0           (XOR с собой = 0)
2. a ^ 0 = a           (XOR с 0 = число)
3. a ^ b = b ^ a       (коммутативность)
4. (a ^ b) ^ c = a ^ (b ^ c)  (ассоциативность)
5. a ^ b ^ b = a       (отмена)
```

### Приложения XOR

```go
// Найти единственное число (остальные встречаются дважды)
func SingleNumber(nums []int) int {
	result := 0
	for _, num := range nums {
		result ^= num
	}
	return result
}

// Найти два единственных числа (остальные встречаются дважды)
func SingleNumberII(nums []int) []int {
	// XOR всех = xor двух единственных
	xor := 0
	for _, num := range nums {
		xor ^= num
	}
	
	// Найти любой установленный бит (где они отличаются)
	rightmostBit := xor & (-xor)
	
	num1, num2 := 0, 0
	for _, num := range nums {
		if num&rightmostBit == 0 {
			num1 ^= num
		} else {
			num2 ^= num
		}
	}
	
	return []int{num1, num2}
}

// Найти пропущенное число в массиве [0, n]
func FindMissing(nums []int) int {
	xor := 0
	for i, num := range nums {
		xor ^= i ^ num
	}
	return xor ^ len(nums)
}

// Проверить если два числа противоположны по знаку
func AreOpposites(a, b int) bool {
	return (a ^ b) < 0 // Если разные знаки, то старший бит отличается
}
```

---

## Bit Manipulation трюки

### Частые паттерны

```go
// 1. Проверить бит на позиции k
isBitSet := (x & (1 << k)) != 0

// 2. Установить бит на позиции k
x |= (1 << k)

// 3. Очистить бит на позиции k
x &= ^(1 << k)

// 4. Переключить бит на позиции k
x ^= (1 << k)

// 5. Получить наименьший установленный бит
lowest := x & (-x)

// 6. Удалить наименьший установленный бит
x &= (x - 1)

// 7. Проверить степень двойки
isPowerOf2 := (x > 0) && ((x & (x-1)) == 0)

// 8. Количество установленных битов
bits := 0
for x > 0 {
	bits += x & 1
	x >>= 1
}

// 9. Позиция наименьшего установленного бита
pos := 0
for (x & 1) == 0 {
	x >>= 1
	pos++
}

// 10. Максимум с той же битовой длиной
maxSameBits := (1 << bits.Len(x)) - 1
```

### Пример: Gray Code

```go
// Преобразовать число в код Грея
func ToGrayCode(n int) int {
	return n ^ (n >> 1)
}

// Преобразовать код Грея в число
func FromGrayCode(gray int) int {
	result := gray
	for gray > 0 {
		gray >>= 1
		result ^= gray
	}
	return result
}
```

---

## Bitmask DP

### Концепция

**Bitmask DP** использует битовую маску для представления подмножества или состояния.

```
Подмножество {1, 3, 4} из {1, 2, 3, 4, 5}:
маска = 01101 (бит 0, 2, 3 установлены)

Преимущества:
- Компактное представление подмножества
- Быстрые операции (добавить, удалить, проверить)
- Может переходить от маски к маске через цикл
```

### Пример 1: Traveling Salesman Problem (TSP)

```go
// Найти кратчайший путь через все города ровно один раз
func TSP(dist [][]int) int {
	n := len(dist)
	INF := math.MaxInt / 2
	
	// dp[mask][i] = кратчайшее расстояние для посещения городов в mask, заканчивая в i
	dp := make([][]int, 1<<n)
	for i := range dp {
		dp[i] = make([]int, n)
		for j := range dp[i] {
			dp[i][j] = INF
		}
	}
	
	// Стартуем из города 0
	dp[1][0] = 0
	
	for mask := 1; mask < (1 << n); mask++ {
		for u := 0; u < n; u++ {
			if dp[mask][u] == INF {
				continue
			}
			
			// Пытаемся посетить город v
			for v := 0; v < n; v++ {
				if (mask & (1 << v)) == 0 { // v не посещён
					newMask := mask | (1 << v)
					dp[newMask][v] = min(dp[newMask][v], dp[mask][u]+dist[u][v])
				}
			}
		}
	}
	
	// Найти минимум вернувшись в город 0
	fullMask := (1 << n) - 1
	ans := INF
	for i := 1; i < n; i++ {
		ans = min(ans, dp[fullMask][i]+dist[i][0])
	}
	
	return ans
}
```

### Пример 2: Subset Sum подсчёт

```go
// Количество подмножеств с определённой суммой
func CountSubsets(nums []int, target int) int {
	n := len(nums)
	
	// dp[mask] = сумма элементов в подмножестве mask
	sums := make([]int, 1<<n)
	for mask := 0; mask < (1 << n); mask++ {
		for i := 0; i < n; i++ {
			if (mask & (1 << i)) != 0 {
				sums[mask] += nums[i]
			}
		}
	}
	
	count := 0
	for mask := 0; mask < (1 << n); mask++ {
		if sums[mask] == target {
			count++
		}
	}
	
	return count
}
```

### Пример 3: Maximum XOR of Two Numbers in an Array

```go
// Найти максимальный XOR двух чисел в массиве
func FindMaximumXOR(nums []int) int {
	maxXor := 0
	prefix := 0
	
	for i := 31; i >= 0; i-- {
		prefix = 0
		prefixes := make(map[int]bool)
		
		for _, num := range nums {
			prefix = (prefix << 1) | ((num >> i) & 1)
			prefixes[prefix] = true
		}
		
		temp := maxXor | (1 << i)
		for p := range prefixes {
			if prefixes[temp^p] {
				maxXor = temp
				break
			}
		}
	}
	
	return maxXor
}
```

---

## Практические примеры

### Пример 1: Number of 1 Bits

```go
// Подсчитать количество установленных битов
func HammingWeight(n uint32) int {
	count := 0
	for n > 0 {
		count += int(n & 1)
		n >>= 1
	}
	return count
}

// Оптимизированная версия
func HammingWeightOptimal(n uint32) int {
	count := 0
	for n > 0 {
		n &= n - 1 // Удаляет наименьший установленный бит
		count++
	}
	return count
}
```

### Пример 2: Reverse Bits

```go
// Развернуть биты числа
func ReverseBits(num uint32) uint32 {
	result := uint32(0)
	for i := 0; i < 32; i++ {
		result = (result << 1) | (num & 1)
		num >>= 1
	}
	return result
}
```

### Пример 3: Bitwise AND of Numbers Range

```go
// Побитовый AND всех чисел в диапазоне [m, n]
func RangeBitwiseAnd(m, n int) int {
	shift := 0
	
	// Найти общую префиксную часть
	for m != n {
		m >>= 1
		n >>= 1
		shift++
	}
	
	return m << shift
}
```

---

## Анализ сложности

### Bitmask DP

| Задача | Состояния | Переходы | Итого |
|--------|-----------|----------|-------|
| **Subset enumeration** | O(2^n) | O(1) | O(2^n) |
| **TSP** | O(2^n * n) | O(n) | O(2^n * n²) |
| **Subset sum** | O(2^n) | O(n) | O(2^n * n) |

### Bit операции

| Операция | Время |
|----------|-------|
| **Все битовые операции** | O(1) |
| **Count bits** | O(log n) |
| **Бинарный поиск на битах** | O(log max_bits) |

---

## Рекомендуемые задачи на LeetCode

- [Single Number](https://leetcode.com/problems/single-number/)
- [Single Number II](https://leetcode.com/problems/single-number-ii/)
- [Number of 1 Bits](https://leetcode.com/problems/number-of-1-bits/)
- [Reverse Bits](https://leetcode.com/problems/reverse-bits/)
- [Bitwise AND of Numbers Range](https://leetcode.com/problems/bitwise-and-of-numbers-range/)
- [Maximum XOR of Two Numbers in an Array](https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/)
- [Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/) (Bitmask)
- [Traveling Salesman Problem](https://leetcode.com/problems/shortest-path-visiting-all-nodes/)

---

## Заключение

1. **Битовые операции** — O(1), очень быстро
2. **XOR свойства** — мощный инструмент для задач с паритетом
3. **Bitmask DP** — экспоненциальная по числу элементов, но эффективен для малых n
4. **Комбинация** — часто используются вместе с DP и поиском

---

**CS 161 — Design of Algorithms**  
*Stanford University, Department of Computer Science*
