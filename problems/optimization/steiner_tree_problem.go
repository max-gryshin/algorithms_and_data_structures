package optimization

import (
	"math"
	"math/rand"
	"sort"
	"time"

	"algorithms_and_data_structures/problems/graphs/dsu/theory"
)

// Point представляет узел на плоскости (например, сервер или филиал)
type Point struct {
	X, Y float64
}

// Edge ребро графа для алгоритма Крускала
type Edge struct {
	U, V   int
	Weight float64
}

// SteinerTournamentSelection выполняет турнирный отбор из популяции
func SteinerTournamentSelection(population []SteinerIndividual) SteinerIndividual {
	best := population[rand.Intn(len(population))]
	for i := 0; i < 2; i++ {
		competitor := population[rand.Intn(len(population))]
		if competitor.Fitness < best.Fitness {
			best = competitor
		}
	}
	return best
}

// SteinerCrossover выполняет одноточечный кроссовер между двумя родителями
func SteinerCrossover(p1, p2 SteinerIndividual) SteinerIndividual {
	childGenes := make([]bool, len(p1.Genes))
	split := rand.Intn(len(p1.Genes))
	for j := 0; j < len(p1.Genes); j++ {
		if j < split {
			childGenes[j] = p1.Genes[j]
		} else {
			childGenes[j] = p2.Genes[j]
		}
	}
	return SteinerIndividual{Genes: childGenes}
}

// SteinerMutate выполняет мутацию особи с заданной вероятностью
func (ind *SteinerIndividual) SteinerMutate(mutationRate float64) {
	for j := 0; j < len(ind.Genes); j++ {
		if rand.Float64() < mutationRate {
			ind.Genes[j] = !ind.Genes[j]
		}
	}
}

// MSTKruskal MST возвращает суммарный вес MST и флаг связности
func MSTKruskal(nodes []Point) (float64, bool) {
	n := len(nodes)
	if n <= 1 {
		return 0, true
	}

	var edges []Edge
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			// Вычисляем евклидово расстояние между узлами (стоимость соединения)
			dist := math.Hypot(nodes[i].X-nodes[j].X, nodes[i].Y-nodes[j].Y)
			edges = append(edges, Edge{U: i, V: j, Weight: dist})
		}
	}

	sort.Slice(edges, func(i, j int) bool {
		return edges[i].Weight < edges[j].Weight
	})

	dsu := theory.NewDSU(n)
	totalWeight := 0.0
	edgeCount := 0

	for _, edge := range edges {
		if dsu.FindSet(edge.U) != dsu.FindSet(edge.V) {
			dsu.Union(edge.U, edge.V)
			totalWeight += edge.Weight
			edgeCount++
			if edgeCount == n-1 {
				break
			}
		}
	}

	if edgeCount != n-1 {
		return math.MaxFloat64, false // Граф несвязен
	}
	return totalWeight, true
}

// Individual особь генетического алгоритма
type SteinerIndividual struct {
	Genes   []bool  // Бинарная маска опциональных узлов
	Fitness float64 // Стоимость сети (MST + аренда узлов)
}

// SteinerTreeResult результат работы алгоритма
type SteinerTreeResult struct {
	Best           SteinerIndividual
	Terminals      []Point
	SteinerPool    []Point
	ActivationCost float64
}

// RunSteinerTreeAlgorithm выполняет гибридный алгоритм GA + MST для задачи Steiner Tree
func RunSteinerTreeAlgorithm(terminals []Point, steinerPool []Point, nodeActivationCost float64, populationSize, generations int, mutationRate float64) SteinerTreeResult {
	rand.Seed(time.Now().UnixNano())

	numSteiner := len(steinerPool)

	// Инициализация популяции
	population := make([]SteinerIndividual, populationSize)
	for i := 0; i < populationSize; i++ {
		genes := make([]bool, numSteiner)
		for j := 0; j < numSteiner; j++ {
			genes[j] = rand.Float64() < 0.5
		}
		population[i] = SteinerIndividual{Genes: genes}
	}

	evaluate := func(ind *SteinerIndividual) {
		// Собираем активный набор узлов для этой особи
		activeNodes := make([]Point, len(terminals))
		copy(activeNodes, terminals)

		activeSteinerCount := 0
		for j, active := range ind.Genes {
			if active {
				activeNodes = append(activeNodes, steinerPool[j])
				activeSteinerCount++
			}
		}

		mstCost, ok := MSTKruskal(activeNodes)
		if !ok {
			ind.Fitness = math.MaxFloat64
			return
		}

		// Общий фитнес = Стоимость линий (MST) + Стоимость аренды выбранных узлов
		ind.Fitness = mstCost + float64(activeSteinerCount)*nodeActivationCost
	}

	// Оцениваем начальную популяцию
	for i := range population {
		evaluate(&population[i])
	}

	// Главный цикл эволюции GA
	for gen := 0; gen < generations; gen++ {
		// Сортируем по возрастанию фитнеса (минимизация)
		sort.Slice(population, func(i, j int) bool {
			return population[i].Fitness < population[j].Fitness
		})

		// Элитизм: сохраняем топ-2 лучших
		newPopulation := make([]SteinerIndividual, populationSize)
		newPopulation[0] = population[0]
		newPopulation[1] = population[1]

		// Размножение
		for i := 2; i < populationSize; i++ {
			// Турнирный отбор родителей
			p1 := SteinerTournamentSelection(population)
			p2 := SteinerTournamentSelection(population)

			// Кроссовер
			child := SteinerCrossover(p1, p2)

			// Мутация
			child.SteinerMutate(mutationRate)

			newPopulation[i] = child
			evaluate(&newPopulation[i])
		}
		population = newPopulation
	}

	// Итоги
	sort.Slice(population, func(i, j int) bool {
		return population[i].Fitness < population[j].Fitness
	})

	return SteinerTreeResult{
		Best:           population[0],
		Terminals:      terminals,
		SteinerPool:    steinerPool,
		ActivationCost: nodeActivationCost,
	}
}
