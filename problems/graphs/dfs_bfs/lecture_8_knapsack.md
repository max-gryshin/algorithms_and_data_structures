# Лекция 8 & 9: The Knapsack Problem

**Курс**: CS 161 — Design of Algorithms  
**Преподаватель**: Кафедра Computer Science, Stanford University  
**Уровень**: Intermediate

---

## Содержание
1. [Введение в задачу о рюкзаке](#введение-в-задачу-о-рюкзаке)
2. [0/1 Knapsack](#01-knapsack)
3. [Unbounded Knapsack](#unbounded-knapsack)
4. [Partition Equal Subset Sum](#partition-equal-subset-sum)
5. [Практические примеры](#практические-примеры)
6. [Анализ сложности](#анализ-сложности)

---

## Введение в задачу о рюкзаке

### Классическая формулировка

**Задача о рюкзаке**: Дан рюкзак с грузоподъёмностью W и набор предметов с весами w₁, w₂, ..., wₙ и стоимостями v₁, v₂, ..., vₙ. Найти подмножество предметов, которое максимизирует стоимость без превышения грузоподъёмности.

```
Рюкзак: W = 50

Предметы:
┌─────┬──────┬─────┐
│ ID  │ Вес  │ Вал │
├─────┼──────┼─────┤
│  1  │  10  │ 60  │
│  2  │  20  │ 100 │
│  3  │  30  │ 120 │
└─────┴──────┴─────┘

Оптимально: взять предметы 1 и 2 (вес=30, ценность=160)
или предметы 2 и 3 (вес=50, ценность=220) ✓
```

### Варианты задачи

```
0/1 Knapsack
└─ Каждый предмет можно взять 0 или 1 раз

Unbounded Knapsack
└─ Каждый предмет можно брать неограниченное количество раз

Bounded Knapsack
└─ Каждый предмет можно брать до k раз

Fractional Knapsack
└─ Можно брать часть предмета (жадный алгоритм)
```

---

## 0/1 Knapsack

### Определение

**0/1 Knapsack** — каждый предмет либо берётся полностью, либо не берётся вообще.

### Рекуррентное соотношение

```
dp[i][w] = максимальная ценность используя первые i предметов 
           с грузоподъёмностью w

dp[i][w] = max(
    dp[i-1][w],                           // не берём i-й предмет
    dp[i-1][w-weight[i]] + value[i]       // берём i-й предмет
)
```

### DP решение O(n·W)

```go
// 0/1 Knapsack за O(n*W) времени и памяти
func KnapsackDP(weights, values []int, capacity int) int {
	n := len(weights)
	
	// dp[i][w] = максимальная ценность для первых i предметов и вместимости w
	dp := make([][]int, n+1)
	for i := range dp {
		dp[i] = make([]int, capacity+1)
	}
	
	for i := 1; i <= n; i++ {
		for w := 1; w <= capacity; w++ {
			// Не берём предмет
			dp[i][w] = dp[i-1][w]
			
			// Если можем взять предмет
			if weights[i-1] <= w {
				dp[i][w] = max(dp[i][w], dp[i-1][w-weights[i-1]]+values[i-1])
			}
		}
	}
	
	return dp[n][capacity]
}

// Восстановить выбранные предметы
func KnapsackItems(weights, values []int, capacity int) []int {
	n := len(weights)
	
	dp := make([][]int, n+1)
	for i := range dp {
		dp[i] = make([]int, capacity+1)
	}
	
	for i := 1; i <= n; i++ {
		for w := 1; w <= capacity; w++ {
			dp[i][w] = dp[i-1][w]
			
			if weights[i-1] <= w {
				dp[i][w] = max(dp[i][w], dp[i-1][w-weights[i-1]]+values[i-1])
			}
		}
	}
	
	// Восстановить выбранные предметы
	selected := []int{}
	w := capacity
	for i := n; i > 0; i-- {
		if dp[i][w] != dp[i-1][w] {
			selected = append(selected, i-1)
			w -= weights[i-1]
		}
	}
	
	return selected
}
```

### Оптимизация памяти до O(W)

```go
// 0/1 Knapsack с оптимизацией памяти до O(W)
func KnapsackSpaceOptimized(weights, values []int, capacity int) int {
	n := len(weights)
	
	// Использовать только один ряд
	dp := make([]int, capacity+1)
	
	for i := 0; i < n; i++ {
		// Идти в обратном порядке чтобы избежать повторного использования
		for w := capacity; w >= weights[i]; w-- {
			dp[w] = max(dp[w], dp[w-weights[i]]+values[i])
		}
	}
	
	return dp[capacity]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```

### DP таблица 0/1 Knapsack

```
weights = [2, 3, 4, 5]
values  = [3, 4, 5, 6]
capacity = 10

       0   1   2   3   4   5   6   7   8   9  10
    0  0   0   0   0   0   0   0   0   0   0   0
    1  0   0   3   3   3   3   3   3   3   3   3
    2  0   0   3   4   4   7   7   7   7   7   7
    3  0   0   3   4   5   7   8   9   9  10  10
    4  0   0   3   4   5   6   8   9  10  10  11

Ответ: 11 (предметы 2 и 3: вес=7, ценность=9+5=14)
```

---

## Unbounded Knapsack

### Определение

**Unbounded Knapsack** — каждый предмет можно использовать неограниченное количество раз.

### Рекуррентное соотношение

```
dp[w] = максимальная ценность при грузоподъёмности w

dp[w] = max(
    dp[w],                                // не берём текущий предмет
    dp[w - weight[i]] + value[i]          // берём текущий предмет (можем взять ещё)
)
```

### DP решение

```go
// Unbounded Knapsack за O(n*W)
func UnboundedKnapsack(weights, values []int, capacity int) int {
	n := len(weights)
	
	// dp[w] = максимальная ценность при вместимости w
	dp := make([]int, capacity+1)
	
	for w := 1; w <= capacity; w++ {
		for i := 0; i < n; i++ {
			if weights[i] <= w {
				// Отличие от 0/1: используем dp[w] текущего ряда, а не предыдущего
				dp[w] = max(dp[w], dp[w-weights[i]]+values[i])
			}
		}
	}
	
	return dp[capacity]
}

// Восстановить выбранные предметы (с повторениями)
func UnboundedKnapsackItems(weights, values []int, capacity int) []int {
	n := len(weights)
	
	dp := make([]int, capacity+1)
	parent := make([]int, capacity+1)
	
	for w := 1; w <= capacity; w++ {
		for i := 0; i < n; i++ {
			if weights[i] <= w && dp[w-weights[i]]+values[i] > dp[w] {
				dp[w] = dp[w-weights[i]] + values[i]
				parent[w] = i
			}
		}
	}
	
	// Восстановить выбранные предметы
	selected := []int{}
	w := capacity
	for w > 0 {
		itemIdx := parent[w]
		selected = append(selected, itemIdx)
		w -= weights[itemIdx]
	}
	
	return selected
}
```

### Разница между 0/1 и Unbounded

```go
// 0/1: идти в ОБРАТНОМ порядке (справа налево)
for w := capacity; w >= weights[i]; w-- {
    dp[w] = max(dp[w], dp[w-weights[i]]+values[i])
}

// Unbounded: идти в ПРЯМОМ порядке (слева направо)
for w := 1; w <= capacity; w++ {
    dp[w] = max(dp[w], dp[w-weights[i]]+values[i])
}
```

---

## Partition Equal Subset Sum

### Определение

**Partition Equal Subset Sum** — разбить массив на два подмножества с равной суммой.

Эквивалентна: найти подмножество с суммой = (total sum)/2

### Решение

```go
// Проверить может ли быть разбиение на две равные части
func CanPartition(nums []int) bool {
	totalSum := 0
	for _, num := range nums {
		totalSum += num
	}
	
	// Если сумма нечётная, невозможно разбить поровну
	if totalSum%2 != 0 {
		return false
	}
	
	target := totalSum / 2
	
	// DP: может ли быть сумма = target
	dp := make([]bool, target+1)
	dp[0] = true
	
	for _, num := range nums {
		// Идти в ОБРАТНОМ порядке (важно!)
		for s := target; s >= num; s-- {
			dp[s] = dp[s] || dp[s-num]
		}
	}
	
	return dp[target]
}

// Найти само разбиение
func FindPartition(nums []int) ([]int, []int) {
	totalSum := 0
	for _, num := range nums {
		totalSum += num
	}
	
	if totalSum%2 != 0 {
		return nil, nil
	}
	
	target := totalSum / 2
	
	// DP с восстановлением
	dp := make([][]bool, len(nums)+1)
	for i := range dp {
		dp[i] = make([]bool, target+1)
		dp[i][0] = true
	}
	
	for i := 1; i <= len(nums); i++ {
		for s := 0; s <= target; s++ {
			dp[i][s] = dp[i-1][s]
			
			if s >= nums[i-1] {
				dp[i][s] = dp[i][s] || dp[i-1][s-nums[i-1]]
			}
		}
	}
	
	if !dp[len(nums)][target] {
		return nil, nil
	}
	
	// Восстановить
	subset1 := []int{}
	s := target
	for i := len(nums); i > 0 && s > 0; i-- {
		if !dp[i-1][s] || (s >= nums[i-1] && dp[i-1][s-nums[i-1]]) {
			if s >= nums[i-1] && dp[i-1][s-nums[i-1]] {
				subset1 = append(subset1, nums[i-1])
				s -= nums[i-1]
			}
		}
	}
	
	subset2 := []int{}
	used := make(map[int]int)
	for _, v := range subset1 {
		used[v]++
	}
	
	for _, num := range nums {
		if used[num] > 0 {
			used[num]--
		} else {
			subset2 = append(subset2, num)
		}
	}
	
	return subset1, subset2
}
```

---

## Практические примеры

### Пример 1: Target Sum

```go
// Найти количество способов достичь целевую сумму
// используя +/- операции перед каждым числом
func FindTargetSumWays(nums []int, target int) int {
	totalSum := 0
	for _, num := range nums {
		totalSum += num
	}
	
	if math.Abs(float64(target)) > float64(totalSum) || (totalSum+target)%2 != 0 {
		return 0
	}
	
	// Преобразовать в subset sum задачу
	// sum(P) - sum(N) = target
	// sum(P) + sum(N) = totalSum
	// => sum(P) = (totalSum + target) / 2
	
	subsetTarget := (totalSum + target) / 2
	
	dp := make([]int, subsetTarget+1)
	dp[0] = 1
	
	for _, num := range nums {
		for s := subsetTarget; s >= num; s-- {
			dp[s] += dp[s-num]
		}
	}
	
	return dp[subsetTarget]
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
```

### Пример 2: Coin Change

```go
// Найти минимальное количество монет для суммы
func CoinChange(coins []int, amount int) int {
	// dp[i] = минимальное количество монет для суммы i
	dp := make([]int, amount+1)
	for i := range dp {
		dp[i] = amount + 1 // "бесконечность"
	}
	dp[0] = 0
	
	for sum := 1; sum <= amount; sum++ {
		for _, coin := range coins {
			if coin <= sum {
				dp[sum] = min(dp[sum], dp[sum-coin]+1)
			}
		}
	}
	
	if dp[amount] > amount {
		return -1
	}
	return dp[amount]
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
```

### Пример 3: Perfect Squares

```go
// Найти минимальное количество полных квадратов для суммы
func NumSquares(n int) int {
	dp := make([]int, n+1)
	for i := range dp {
		dp[i] = i // максимум i*1
	}
	
	for i := 1; i <= n; i++ {
		for j := 1; j*j <= i; j++ {
			dp[i] = min(dp[i], dp[i-j*j]+1)
		}
	}
	
	return dp[n]
}
```

---

## Анализ сложности

| Вариант | Время | Память | Примечание |
|---------|-------|--------|-----------|
| **0/1 Knapsack** | O(n·W) | O(n·W) | Каждый один раз |
| **0/1 опт.** | O(n·W) | O(W) | Один ряд DP |
| **Unbounded** | O(n·W) | O(W) | Каждый много раз |
| **Partition** | O(n·sum/2) | O(sum/2) | Subset sum |

---

## DP таблица примеры

### 0/1 Knapsack
```
weights=[1,2,3,4], values=[10,20,30,40], capacity=5

        0   1   2   3   4   5
    0   0   0   0   0   0   0
    1  (0) 10  10  10  10  10
    2  (0) 10  20  30  30  30
    3  (0) 10  20  30  40  50
    4  (0) 10  20  30  40  50

Ответ: 50
```

### Unbounded Knapsack
```
weights=[1,3,4], values=[1,3,2], capacity=6

dp[0]=0, dp[1]=1, dp[2]=2, dp[3]=3, dp[4]=3, dp[5]=4, dp[6]=4
```

---

## Рекомендуемые задачи на LeetCode

- [0/1 Knapsack](https://leetcode.com/problems/partition-equal-subset-sum/) — классический
- [Coin Change](https://leetcode.com/problems/coin-change/) — неограниченный
- [Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/) — разбиение
- [Target Sum](https://leetcode.com/problems/target-sum/) — +/- комбинации
- [Perfect Squares](https://leetcode.com/problems/perfect-squares/) — сумма квадратов
- [Coin Change II](https://leetcode.com/problems/coin-change-2/) — количество способов
- [House Robber](https://leetcode.com/problems/house-robber/) — условное выбирание

---

## Заключение

Задача о рюкзаке — одна из фундаментальных задач DP:

1. **0/1 Knapsack** — классическая NP-полная задача
2. **Unbounded** — частозапрашиваемая в интервью
3. **Partition Equal Subset Sum** — проверка разбиения
4. **Приложения** — распределение ресурсов, максимизация прибыли

Ключевое понимание:
- **Обратный порядок** в 0/1 (избегаем повторного использования)
- **Прямой порядок** в Unbounded (позволяем повторное использование)

---

**CS 161 — Design of Algorithms**  
*Stanford University, Department of Computer Science*
