# CS 161 — Design of Algorithms: Полный курс лекций

**Stanford University, Department of Computer Science**

Этот набор лекций охватывает ключевые алгоритмы проектирования и структуры данных, необходимые для любого инженера.

---

## 📚 Содержание курса

### [Лекция 0: Стеки, Очереди и Основные Структуры Данных](lecture_0_stacks_queues.md)
**Фундаментальные структуры данных**

Краткое содержание:
- Стеки (LIFO) — реализация и применение
- Очереди (FIFO) — циркулярная буферизация
- Deque — двусторонняя очередь
- Приложения: DFS, BFS, LRU Cache, Sliding Window
- Примеры: Valid Parentheses, Trapping Rain Water, Daily Temperatures

**Рекомендуемые задачи**: Valid Parentheses, Daily Temperatures, Trapping Rain Water, Sliding Window Maximum, LRU Cache

---

### [Лекция 1: BFS, Дейкстра и Основы Теории Графов](theory/lecture_1_bfs_dijkstra.md)
**Основы и поиск кратчайшего пути**

Краткое содержание:
- Введение в графы (ориентированные, неориентированные, взвешенные)
- Представление графов (список смежности vs матрица)
- Breadth-First Search (BFS) для невзвешенных графов
- Поиск кратчайшего пути в невзвешенных графах
- Алгоритм Дейкстры для взвешенных графов
- Практические примеры: Shortest Path in Binary Matrix, Network Delay Time
- Анализ сложности и когда использовать каждый метод

**Рекомендуемые задачи**: Number of Islands, Shortest Path in Binary Matrix, Network Delay Time, Path With Minimum Effort, Word Ladder

---

### [Лекция 2: Теория Игр и Minimax](lecture_2_game_theory.md)
**Игры с нулевой суммой и стратегическое взаимодействие**

Краткое содержание:
- Введение в теорию игр и комбинаторные игры
- Игры с нулевой суммой и полная информация
- Алгоритм Minimax для поиска оптимального хода
- Alpha-Beta Pruning для оптимизации
- Динамическое программирование в играх
- Stone Game — задача с интервальным DP
- Nim Game и теорема Спрага-Гранди
- Практические примеры: угадывание чисел, Stone Game II

**Рекомендуемые задачи**: Stone Game I–IX, Nim Game, Can I Win, Guess Number Higher or Lower II, Predict the Winner

---

### [Лекция 3: DFS, Связность и Топологическая Сортировка](theory/lecture_3_dfs.md)
**Глубинный поиск и приложения**

Краткое содержание:
- Depth-First Search (DFS) — рекурсивная и итеративная реализация
- Поиск связных компонент в графе
- Обход деревьев (Pre-order, In-order, Post-order)
- Топологическая сортировка (алгоритм DFS и Kahn)
- Обнаружение циклов в ориентированных графах
- Практические примеры: Number of Islands, Course Schedule, Max Area of Island

**Рекомендуемые задачи**: Number of Islands, Max Area of Island, Course Schedule, Course Schedule II, Inorder Traversal, Detect Cycle, Surrounded Regions

---

### [Лекция 4: Мосты, Точки Артикуляции и Алгоритмы Тарджана/Косарайю](theory/lecture_4_bridges_articulation.md)
**Продвинутый анализ графов (Hard)**

Краткое содержание:
- Точки артикуляции (критичные вершины)
- Мосты в графе (критичные рёбра)
- Сильно связные компоненты (SCC)
- Алгоритм Тарджана для SCC (один проход)
- Алгоритм Косарайю для SCC (два прохода)
- Практические примеры: Critical Connections in a Network
- Сравнение методов

**Рекомендуемые задачи**: Critical Connections in a Network, Number of Connected Components, Redundant Connection, Accounts Merge

---

### [Лекция 5 & 9: Disjoint Set Union (СНМ / DSU)](../dsu/theory/lecture.md)
**Система непересекающихся множеств**

Краткое содержание:
- Введение в DSU и операции Find/Union
- Базовая реализация
- Оптимизация: Path Compression и Union by Rank
- Union by Size как альтернатива
- Анализ сложности (обратная функция Аккермана)
- Применение в задачах: Number of Provinces, Redundant Connection
- Accounts Merge и другие примеры
- Алгоритм Краскала для MST с DSU

**Рекомендуемые задачи**: Number of Provinces, Redundant Connection, Accounts Merge, Friend Circles, Couples Holding Hands, Smallest String With Swaps

---

### [Лекция 6: Бинарный поиск и Two Pointers](lecture_6_binary_search.md)
**Классический поиск и стратегические указатели**

Краткое содержание:
- Классический бинарный поиск на отсортированном массиве
- Варианты: leftmost, rightmost, диапазон
- Answer Search — бинпоиск на пространстве ответов
- Two Pointers паттерн (встречающиеся, медленный/быстрый)
- Sliding Window для substring задач
- Практические примеры: Search in Rotated Array, 3Sum, Container with Most Water

**Рекомендуемые задачи**: Binary Search, Search in Rotated Sorted Array, Find First and Last Position, Container With Most Water, Remove Duplicates, 3Sum, Longest Substring Without Repeating

---

### [Лекция 6a: Binary Search Tree и AVL Tree](lecture_6a_bst_avl.md)
**Деревья поиска и балансировка**

Краткое содержание:
- Binary Search Tree (BST) — вставка, удаление, поиск
- Проблема дисбаланса (skewed tree)
- Ротации для восстановления баланса (LL, RR, LR, RL)
- AVL Tree — самобалансирующееся дерево
- Анализ сложности (худший vs средний)
- Примеры: Validate BST, Kth Smallest, Closest Value

**Рекомендуемые задачи**: Validate Binary Search Tree, Kth Smallest Element, Lowest Common Ancestor of BST, Binary Search Tree Iterator, Closest Binary Search Tree Value, Convert Sorted Array to BST, Balanced Binary Tree

---

### [Лекция 6b: Segment Tree и Fenwick Tree](lecture_6b_segment_fenwick.md)
**Range Queries и структуры данных для интервалов**

Краткое содержание:
- Range Query задачи (сумма, минимум, максимум на диапазоне)
- Segment Tree с O(log n) query и update
- Lazy Propagation для range updates
- Fenwick Tree (Binary Indexed Tree) — компактнее, проще
- Координатная компрессия для больших значений
- Практические примеры: Range Sum Query, Counting Smaller Numbers, 2D queries

**Рекомендуемые задачи**: Range Sum Query Mutable, Range Sum Query 2D Mutable, Counting Smaller Numbers After Self, Reverse Pairs, Segment Tree Modification

---

### [Лекция 6c: Tree DP (Dynamic Programming на деревьях)](lecture_6c_tree_dp.md)
**DP на иерархических структурах**

Краткое содержание:
- Tree DP vs обычное DP
- Bottom-Up подход (дети перед родителем)
- Two-State DP (с корнем / без корня)
- Мемоизация на деревьях
- Примеры: Maximum Path Sum, Diameter, House Robber III
- Advanced: Binary Tree Cameras, Distribute Coins, LCA

**Рекомендуемые задачи**: Maximum Path Sum, Diameter of Binary Tree, Balanced Binary Tree, House Robber III, Lowest Common Ancestor, Binary Tree Cameras, Distribute Coins in Binary Tree

---

### [Лекция 7: LIS и LCS (НВП и НОП)](lecture_7_lis_lcs.md)
**Классическое динамическое программирование**

Краткое содержание:
- Динамическое программирование: принцип и свойства
- Longest Increasing Subsequence (LIS) — DP O(n²) и бинпоиск O(n log n)
- Longest Common Subsequence (LCS) для двух строк
- Восстановление самой последовательности
- Edit Distance (Levenstein) для близости строк
- Distinct Subsequences — подсчёт вариантов
- Interleaving String — проверка интерлейвинга

**Рекомендуемые задачи**: Longest Increasing Subsequence, Russian Doll Envelopes, Longest Common Subsequence, Edit Distance, Distinct Subsequences, Interleaving String

---

### [Лекция 8 & 9: Задача о рюкзаке (Knapsack)](lecture_8_knapsack.md)
**Вариации и оптимизации**

Краткое содержание:
- Классическая формулировка задачи о рюкзаке
- 0/1 Knapsack с DP O(n·W)
- Оптимизация памяти до O(W)
- Восстановление выбранных предметов
- Unbounded Knapsack — неограниченное количество
- Partition Equal Subset Sum — разбиение на равные части
- Target Sum — +/- комбинации для достижения целевой суммы
- Coin Change и Perfect Squares

**Рекомендуемые задачи**: 0/1 Knapsack, Coin Change, Partition Equal Subset Sum, Target Sum, Perfect Squares, Coin Change II, House Robber

---

### [Лекция 9: Minimum Spanning Tree (Краскал и Прим)](lecture_9_mst.md)
**Остовные деревья и их применение**

Краткое содержание:
- Определение и свойства MST
- Алгоритм Краскала (жадный на рёбрах)
  - Сортировка рёбер
  - Использование DSU
  - Обнаружение циклов
- Алгоритм Прима (жадный на вершинах)
  - Приоритетная очередь
  - Постепенное расширение дерева
- Сравнение методов (когда какой использовать)
- Лемма о разрезе (доказательство корректности)
- Практические примеры: Min Cost to Connect All Points

**Рекомендуемые задачи**: Min Cost to Connect All Points, Redundant Connection, Connecting Cities With Minimum Cost, Number of Provinces

---

### [Лекция 12: Maximum Flow и Min Cut](lecture_12_maximum_flow.md)
**Потоки и сетевые задачи**

Краткое содержание:
- Flow Network — граф с пропускными способностями
- Ford-Fulkerson алгоритм (увеличивающие пути)
- Edmonds-Karp алгоритм (BFS для стабильности)
- Dinic алгоритм (level graph и blocking flows)
- Min Cut = Max Flow (теорема)
- Приложения: паросочетание в двудольных графах
- Сравнение алгоритмов по скорости

**Применение**: паросочетание, вырезание графа, распределение ресурсов

---

### [Лекция 13: Bit Manipulation и Bit DP](lecture_13_bit_manipulation.md)
**Битовые операции и динамическое программирование на битах**

Краткое содержание:
- Базовые битовые операции (AND, OR, XOR, NOT, сдвиги)
- Проверка, установка, очистка битов
- XOR свойства (отмена, коммутативность)
- Поиск единственного элемента через XOR
- Bitmask DP — представление подмножеств
- Traveling Salesman Problem (TSP) через bitmask
- Максимум XOR в массиве
- Трюки: степень двойки, подсчёт битов, развёртывание

**Рекомендуемые задачи**: Single Number, Number of 1 Bits, Reverse Bits, Maximum XOR, Bitwise AND Range, Partition Equal Subset Sum (bitmask)

---

### [Лекция 11: Кратчайшие пути (Продолжение)](lecture_11_shortest_paths.md)
**Продвинутые алгоритмы для всех пар**

Краткое содержание:
- Обзор всех алгоритмов кратчайшего пути
- Беллман-Форд для отрицательных весов
  - V-1 релаксаций
  - Обнаружение отрицательных циклов
  - Восстановление пути
- Флойд-Уоршелл для всех пар
  - DP подход с промежуточными вершинами
  - Восстановление пути через Next матрицу
  - Обнаружение отрицательных циклов
- SPFA — улучшенный Беллман-Форд на практике
- Таблица выбора алгоритма
- Практические примеры: Cheapest Flights Within K Stops

**Рекомендуемые задачи**: Network Delay Time, Cheapest Flights Within K Stops, Path With Minimum Effort, Shortest Path in a Grid with Obstacles Elimination, Minimum Cost to Make at Least One Valid Path

---

## 🎯 Дополнительные материалы

### [SCC Визуализация Косарайю](theory/scc_kosaraju_visualization.md)
Диаграммы и пошаговое объяснение алгоритма Косарайю на практическом примере.

---

## 📊 Рекомендуемый порядок изучения

### Foundation (Основы):
1. **Лекция 0** — Стеки и очереди (используются везде)
2. **Лекция 6a** — BST / AVL (базовые структуры поиска)
3. **Лекция 6** — Бинарный поиск (фундаментальный паттерн)

### Core Algorithms (Главные алгоритмы):
4. **Лекция 1** (BFS, Дейкстра) — основы графов
5. **Лекция 3** (DFS) — обход и компоненты
6. **Лекция 6b** (Segment/Fenwick Tree) — range queries
7. **Лекция 6c** (Tree DP) — DP на структурах
8. **Лекция 5 & 9** (DSU) — объединение множеств

### Advanced Algorithms (Продвинутые):
9. **Лекция 7** (LIS, LCS) — последовательности
10. **Лекция 8** (Knapsack) — DP на ограничениях
11. **Лекция 9** (MST) — остовные деревья
12. **Лекция 13** (Bit Manipulation) — битовые паттерны
13. **Лекция 11** (Floyd-Warshall, Bellman-Ford) — все пары путей

### Expert (Специализированные):
14. **Лекция 4** (SCC, Мосты) — критичные элементы графов
15. **Лекция 2** (Теория игр) — стратегические задачи
16. **Лекция 12** (Maximum Flow) — потоки и паросочетания

---

## 💻 Примеры кода

Все примеры кода в лекциях написаны на **Go** и готовы к использованию:
- Полные реализации алгоритмов
- Функции для восстановления решения
- Обработка граничных случаев
- Оптимизированные версии где возможно

---

## ⏱️ Сложность алгоритмов

### По времени выполнения

| Алгоритм | Время | Применение |
|----------|-------|-----------|
| **Структуры данных** | |
| Stack/Queue | O(1) | LIFO/FIFO операции |
| Binary Search | O(log n) | На отсортированном массиве |
| BST операция | O(log n) avg | Поиск, вставка, удаление |
| AVL Tree операция | O(log n) | Гарантированная балансировка |
| Segment Tree | O(log n) | Range queries |
| Fenwick Tree | O(log n) | Prefix sums, range queries |
| **Графы** | |
| BFS | O(V + E) | Невзвешенные, кратчайший путь |
| DFS | O(V + E) | Обход, циклы, компоненты |
| Дейкстра | O((V+E) log V) | Неотрицательные веса |
| Беллман-Форд | O(V·E) | Отрицательные веса |
| Флойд-Уоршелл | O(V³) | Все пары кратчайших путей |
| Краскал | O(E log E) | MST, разреженные графы |
| Прим | O(E log V) | MST, плотные графы |
| **Максимальный поток** | |
| Ford-Fulkerson | O(E·maxFlow) | Маленькие графы |
| Edmonds-Karp | O(V·E²) | Стабильный выбор |
| Dinic | O(V²·E) | Большие графы |
| **Dynamic Programming** | |
| LIS (DP) | O(n²) | Подпоследовательность |
| LIS (бинпоиск) | O(n log n) | Оптимальное решение |
| LCS | O(m·n) | Две строки |
| Knapsack | O(n·W) | Рюкзак, подсчёт способов |
| Tree DP | O(n) | На деревьях (dfs) |
| Bitmask DP | O(2^n·n) | Подмножества |
| **Специальные** | |
| DSU | O(α(n)) | Объединение множеств |
| XOR операции | O(1) | Битовые манипуляции |

---

## 📝 Как использовать эти лекции

1. **Прочитайте** введение и основные концепции
2. **Изучите** реализацию на Go
3. **Поймите** анализ сложности
4. **Решите** рекомендуемые задачи на LeetCode
5. **Повторите** на других языках если нужно

---

## 🔗 Внешние ресурсы

- **LeetCode**: https://leetcode.com — практика задач
- **Visualgo**: https://visualgo.net — визуализация алгоритмов
- **CLRS**: "Introduction to Algorithms" — классический учебник
- **Competitive Programming**: cp-algorithms.com — детальные описания

---

## 📄 Лицензия и использование

Эти лекции созданы в стиле курса CS 161 (Design of Algorithms) в Stanford University.
Используются в образовательных целях.

---

**Последнее обновление**: 19 сентября 2026

Все примеры кода готовы к запуску и могут быть использованы для обучения и собеседований.
