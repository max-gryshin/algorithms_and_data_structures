# Лекция 2: Теория Игр и Minimax

**Курс**: CS 161 — Design of Algorithms  
**Преподаватель**: Кафедра Computer Science, Stanford University  
**Уровень**: Intermediate

---

## Содержание
1. [Введение в теорию игр](#введение-в-теорию-игр)
2. [Игры с нулевой суммой](#игры-с-нулевой-суммой)
3. [Minimax алгоритм](#minimax-алгоритм)
4. [Динамическое программирование в играх](#динамическое-программирование-в-играх)
5. [Stone Game](#stone-game)
6. [Nim Game](#nim-game)
7. [Практические примеры](#практические-примеры)
8. [Анализ сложности](#анализ-сложности)

---

## Введение в теорию игр

### Что такое теория игр?

Теория игр — это математическое исследование стратегического взаимодействия между рациональными агентами. В Computer Science мы сосредотачиваемся на **комбинаторных играх**:

- **Двое игроков** (не более)
- **Полная информация** (оба видят состояние игры)
- **Детерминированность** (нет случайности)
- **Беспристрастность** (равные правила для обоих)
- **Конечность** (игра всегда заканчивается)

### Классификация позиций

```
Position (позиция)
    ├── Winning (выигрышная)
    │   └── существует ход в проигрышную позицию
    │
    └── Losing (проигрышная)
        └── все ходы ведут в выигрышные позиции
```

---

## Игры с нулевой суммой

### Определение

**Игра с нулевой суммой** — это игра, где прибыль одного игрока равна убытку другого:

```
Выигрыш игрока 1 + Выигрыш игрока 2 = 0

или

Выигрыш игрока 1 = -Выигрыш игрока 2
```

### Свойства

| Свойство | Описание |
|----------|---------|
| **Полная противоположность интересов** | Что хорошо для одного — плохо для другого |
| **Рациональность обоих игроков** | Оба играют оптимально |
| **Теорема фон Неймана** | Существует стратегия гарантирующая минимальный выигрыш |

### Пример: Крестики-нолики

```
Если оба играют оптимально → Ничья
```

---

## Minimax алгоритм

### Концепция

Minimax — это алгоритм нахождения оптимального хода путём:

1. **Maximizer** (максимизирующий игрок) ищет максимальный выигрыш
2. **Minimizer** (минимизирующий игрок) ищет минимум выигрыша Maximizer'а

```
          Maximizer (MAX)
                  |
           ┌──────┼──────┐
           |      |      |
         MIN    MIN    MIN
         /|\    /|\    /|\
        L W D  L W D  L W D
```

### Псевдокод

```
minimax(state, depth, isMaximizer):
    if terminal(state):
        return evaluate(state)
    
    if isMaximizer:
        maxEval = -∞
        for each move:
            eval = minimax(nextState, depth+1, False)
            maxEval = max(maxEval, eval)
        return maxEval
    else:
        minEval = +∞
        for each move:
            eval = minimax(nextState, depth+1, True)
            minEval = min(minEval, eval)
        return minEval
```

### Реализация на Go

```go
// Minimax алгоритм для игры с нулевой суммой
func Minimax(state interface{}, depth int, isMaximizer bool, evaluator func(interface{}) int) int {
	if depth == 0 || isTerminal(state) {
		return evaluator(state)
	}
	
	moves := getMoves(state)
	
	if isMaximizer {
		maxEval := math.MinInt
		for _, move := range moves {
			nextState := applyMove(state, move)
			eval := Minimax(nextState, depth-1, false, evaluator)
			if eval > maxEval {
				maxEval = eval
			}
		}
		return maxEval
	} else {
		minEval := math.MaxInt
		for _, move := range moves {
			nextState := applyMove(state, move)
			eval := Minimax(nextState, depth-1, true, evaluator)
			if eval < minEval {
				minEval = eval
			}
		}
		return minEval
	}
}
```

### Alpha-Beta Pruning (оптимизация)

Minimax неэффективен, так как исследует все возможные ходы. **Alpha-Beta Pruning** отсекает ветви, которые не могут повлиять на результат:

```
α (alpha) = лучший результат для Maximizer
β (beta)  = лучший результат для Minimizer

Если α ≥ β → обрезать ветку (pruning)
```

```go
// Minimax с Alpha-Beta Pruning
func MinimaxAB(state interface{}, depth int, alpha, beta int, isMaximizer bool, evaluator func(interface{}) int) int {
	if depth == 0 || isTerminal(state) {
		return evaluator(state)
	}
	
	moves := getMoves(state)
	
	if isMaximizer {
		maxEval := math.MinInt
		for _, move := range moves {
			nextState := applyMove(state, move)
			eval := MinimaxAB(nextState, depth-1, alpha, beta, false, evaluator)
			if eval > maxEval {
				maxEval = eval
			}
			if eval > alpha {
				alpha = eval
			}
			if alpha >= beta {
				break // Beta cutoff
			}
		}
		return maxEval
	} else {
		minEval := math.MaxInt
		for _, move := range moves {
			nextState := applyMove(state, move)
			eval := MinimaxAB(nextState, depth-1, alpha, beta, true, evaluator)
			if eval < minEval {
				minEval = eval
			}
			if eval < beta {
				beta = eval
			}
			if alpha >= beta {
				break // Alpha cutoff
			}
		}
		return minEval
	}
}
```

---

## Динамическое программирование в играх

### Идея

Многие игры имеют **перекрывающиеся подзадачи** — одно и то же состояние игры может быть достигнуто несколькими путями. Используем **мемоизацию**:

```go
// Minimax с мемоизацией
func MinimaxMemo(state interface{}, depth int, isMaximizer bool, evaluator func(interface{}) int, memo map[string]int) int {
	stateStr := stateToString(state) // Преобразование состояния в ключ
	
	if val, exists := memo[stateStr]; exists {
		return val
	}
	
	if depth == 0 || isTerminal(state) {
		return evaluator(state)
	}
	
	moves := getMoves(state)
	var result int
	
	if isMaximizer {
		result = math.MinInt
		for _, move := range moves {
			nextState := applyMove(state, move)
			eval := MinimaxMemo(nextState, depth-1, false, evaluator, memo)
			if eval > result {
				result = eval
			}
		}
	} else {
		result = math.MaxInt
		for _, move := range moves {
			nextState := applyMove(state, move)
			eval := MinimaxMemo(nextState, depth-1, true, evaluator, memo)
			if eval < result {
				result = eval
			}
		}
	}
	
	memo[stateStr] = result
	return result
}
```

---

## Stone Game

### Задача

Две игроки A и B выбирают камни из массива. В каждый ход игрок может выбрать камень слева или справа из оставшихся. Каждый получает сумму своих камней. Оба играют оптимально.

```
stones = [3, 7, 4, 5]

A выбирает 3 или 5?
  - Если 3: остаётся [7, 4, 5] для B
  - Если 5: остаётся [3, 7, 4] для B

B выбирает оптимально...
```

### Решение: DP с интервалом

```go
// Stone Game - найти максимальную разницу в очках для текущего игрока
func StoneGameDP(stones []int) int {
	n := len(stones)
	
	// dp[i][j] = максимальная разница (текущий игрок - противник)
	// когда игроки выбирают из stones[i:j+1]
	dp := make([][]int, n)
	for i := range dp {
		dp[i] = make([]int, n)
	}
	
	// Базовый случай: один камень
	for i := 0; i < n; i++ {
		dp[i][i] = stones[i]
	}
	
	// Заполнять по длине интервала
	for len := 2; len <= n; len++ {
		for i := 0; i <= n-len; i++ {
			j := i + len - 1
			// Выбираем камень слева или справа
			// Если выберем i: получим stones[i] + оставшееся
			// но противник будет максимизировать, значит нам вычтут его максимум
			dp[i][j] = max(
				stones[i]-dp[i+1][j],
				stones[j]-dp[i][j-1],
			)
		}
	}
	
	return dp[0][n-1]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// AliceWins возвращает может ли Alice выиграть
func AliceWins(stones []int) bool {
	totalSum := 0
	for _, s := range stones {
		totalSum += s
	}
	aliceDiff := StoneGameDP(stones)
	aliceSum := (totalSum + aliceDiff) / 2
	return aliceSum > totalSum-aliceSum
}
```

### Сложность

| Метрика | Значение |
|---------|----------|
| **Время** | O(n²) |
| **Память** | O(n²) |
| **Состояния** | O(n²) |

---

## Nim Game

### Правила

- Есть N кучек камней с количествами a₁, a₂, ..., aₙ
- Игроки чередуются, беря любое количество камней из одной кучки
- Берущий последний камень выигрывает
- Найти выигрышную стратегию

### Теорема Спрага-Гранди

**Выигрышная позиция ⟺ XOR всех размеров кучек = 0**

```
stones = [1, 4, 5]
XOR: 1 ⊕ 4 ⊕ 5 = 0 ⟹ Текущий игрок проигрывает (если противник играет оптимально)

stones = [1, 3, 5]
XOR: 1 ⊕ 3 ⊕ 5 = 7 ≠ 0 ⟹ Текущий игрок выигрывает
```

### Решение на Go

```go
// Nim Game - может ли текущий игрок выиграть?
func CanWinNim(piles []int) bool {
	xorSum := 0
	for _, pile := range piles {
		xorSum ^= pile
	}
	return xorSum != 0
}

// Найти выигрышный ход в Nim Game
func FindWinningMove(piles []int) (pileIndex, newSize int, found bool) {
	xorSum := 0
	for _, pile := range piles {
		xorSum ^= pile
	}
	
	if xorSum == 0 {
		return -1, -1, false // Текущая позиция проигрышная
	}
	
	// Найти кучку, которую можно уменьшить
	for i, pile := range piles {
		// После удаления камней из pile[i], XOR должен стать 0
		newPile := pile ^ xorSum
		if newPile < pile {
			return i, newPile, true
		}
	}
	
	return -1, -1, false
}

// Пример
func main() {
	piles := []int{1, 4, 5}
	canWin := CanWinNim(piles) // false, это проигрышная позиция
	
	piles2 := []int{1, 3, 5}
	canWin2 := CanWinNim(piles2) // true
	i, newSize, _ := FindWinningMove(piles2)
	// Результат: уменьшить piles2[i] до newSize
}
```

### Почему XOR работает?

```
Свойство XOR:
- a ⊕ a = 0
- a ⊕ 0 = a
- XOR коммутативен и ассоциативен

Стратегия: всегда оставляй противнику позицию где XOR = 0
```

---

## Практические примеры

### Пример 1: Guess Number Higher or Lower Game

```go
// Угадать число, оптимально минимизируя стоимость
func GuessNumber(n int) int {
	memo := make(map[int]int)
	
	var minCost func(int, int) int
	minCost = func(low, high int) int {
		if low >= high {
			return 0
		}
		
		key := low*10000 + high
		if val, exists := memo[key]; exists {
			return val
		}
		
		result := math.MaxInt
		for i := low; i <= high; i++ {
			// Худший случай: угадываем i
			cost := i + max(minCost(low, i-1), minCost(i+1, high))
			if cost < result {
				result = cost
			}
		}
		
		memo[key] = result
		return result
	}
	
	return minCost(1, n)
}
```

### Пример 2: Predict the Winner (Stone Game II)

```go
// Может ли текущий игрок выиграть в Stone Game II?
func PredictTheWinner(nums []int) bool {
	memo := make(map[int]int)
	
	var maxDiff func(int, int, int) int
	maxDiff = func(start, end, multiplier int) int {
		if start > end {
			return 0
		}
		
		key := start*1000 + end*1000000 + multiplier
		if val, exists := memo[key]; exists {
			return val
		}
		
		// Выбираем от 1 до min(2*M, end-start+1) камней
		m := (end - start + 2) / 2
		result := math.MinInt
		
		for x := 1; x <= m && start+x-1 <= end; x++ {
			sum := 0
			for i := 0; i < x; i++ {
				sum += nums[start+i]
			}
			remaining := maxDiff(start+x, end, max(m, x))
			if sum*multiplier-remaining > result {
				result = sum*multiplier - remaining
			}
		}
		
		memo[key] = result
		return result
	}
	
	return maxDiff(0, len(nums)-1, 1) >= 0
}
```

---

## Анализ сложности

| Алгоритм | Время | Память | Применение |
|----------|-------|--------|-----------|
| **Minimax** | O(b^d) | O(d) | Маленькие игры |
| **Alpha-Beta Pruning** | O(b^(d/2)) | O(d) | Большие поиск-пространства |
| **DP в играх** | O(n²) или O(n³) | O(n²) | Интервальные игры |
| **Nim Game (XOR)** | O(n) | O(1) | Кучки камней |

Где:
- b = branching factor (среднее число ходов)
- d = depth (глубина поиска)

---

## Рекомендуемые задачи на LeetCode

- [Stone Game](https://leetcode.com/problems/stone-game/) — DP интервалом
- [Stone Game II](https://leetcode.com/problems/stone-game-ii/) — вариант с ограничением
- [Stone Game III](https://leetcode.com/problems/stone-game-iii/) — три камня за ход
- [Can I Win](https://leetcode.com/problems/can-i-win/) — Minimax с мемоизацией
- [Guess Number Higher or Lower II](https://leetcode.com/problems/guess-number-higher-or-lower-ii/) — минимакс-стоимость
- [Predict the Winner](https://leetcode.com/problems/predict-the-winner/) — Stone Game II вариант
- [Nim Game](https://leetcode.com/problems/nim-game/) — классическая игра

---

## Заключение

Теория игр в Computer Science касается не только самих игр, но и:

- **Аукционов и торговли** — выигрышные стратегии
- **Сетевых протоколов** — равновесия между узлами
- **Криптографии** — игры между криптографом и взломщиком
- **Искусственного интеллекта** — AlphaGo использует варианты этих идей

Ключевые идеи:
1. **Minimax** — поиск оптимального хода
2. **Alpha-Beta Pruning** — оптимизация Minimax
3. **DP в играх** — мемоизация состояний
4. **Теория Спрага-Гранди** — анализ беспристрастных игр

---

**CS 161 — Design of Algorithms**  
*Stanford University, Department of Computer Science*
