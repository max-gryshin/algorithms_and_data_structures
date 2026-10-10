package optimization

import (
	"math"
	"math/rand"
	"time"
)

// Город с координатами на плоскости
type City struct {
	X, Y float64
}

// Вычисление евклидова расстояния между городами
func (c City) DistanceTo(other City) float64 {
	dx := c.X - other.X
	dy := c.Y - other.Y
	return math.Sqrt(dx*dx + dy*dy)
}

const (
	numCities         = 10
	TSPpopSize        = 80
	TSPmutationRate   = 0.1
	TSPmaxGenerations = 300
)

var cities []City

type TSPIndividual struct {
	Route   []int
	Fitness float64 // Чем меньше расстояние, тем лучше (но в фитнесе инвертируем: 1 / distance)
}

func initCities() {
	// Фиксированные координаты 10 точек
	cities = []City{
		{X: 60, Y: 200}, {X: 180, Y: 200}, {X: 80, Y: 180},
		{X: 140, Y: 180}, {X: 20, Y: 160}, {X: 100, Y: 160},
		{X: 200, Y: 160}, {X: 140, Y: 140}, {X: 40, Y: 120},
		{X: 100, Y: 120},
	}
}

// Оценка маршрута (общая длина)
func (ind *TSPIndividual) Evaluate() {
	totalDist := 0.0
	for i := 0; i < len(ind.Route); i++ {
		from := cities[ind.Route[i]]
		to := cities[ind.Route[(i+1)%len(ind.Route)]] // замыкаем кольцо
		totalDist += from.DistanceTo(to)
	}
	// Обратная величина, так как ГА максимизирует фитнес
	ind.Fitness = 1.0 / totalDist
}

// Упорядоченный кроссовер (Ordered Crossover - OX)
func OrderedCrossover(p1, p2 []int) []int {
	length := len(p1)
	child := make([]int, length)
	for i := range child {
		child[i] = -1
	}

	// Выбираем случайный диапазон
	start := rand.Intn(length)
	end := rand.Intn(length)
	if start > end {
		start, end = end, start
	}

	// Копируем отрезок от первого родителя
	for i := start; i <= end; i++ {
		child[i] = p1[i]
	}

	// Заполняем оставшиеся места генами второго родителя
	p2Idx := 0
	for i := 0; i < length; i++ {
		if child[i] == -1 {
			// Ищем элемент из p2, которого еще нет у ребенка
			for {
				candidate := p2[p2Idx]
				p2Idx = (p2Idx + 1) % length
				exists := false
				for _, val := range child {
					if val == candidate {
						exists = true
						break
					}
				}
				if !exists {
					child[i] = candidate
					break
				}
			}
		}
	}
	return child
}

// Мутация свапом (обмен двух случайных городов местами)
func (ind *TSPIndividual) Mutate() {
	if rand.Float64() < TSPmutationRate {
		i := rand.Intn(len(ind.Route))
		j := rand.Intn(len(ind.Route))
		ind.Route[i], ind.Route[j] = ind.Route[j], ind.Route[i]
	}
}

func TSPTournamentSelection(pop []TSPIndividual) TSPIndividual {
	best := pop[rand.Intn(len(pop))]
	for i := 0; i < 2; i++ {
		cand := pop[rand.Intn(len(pop))]
		if cand.Fitness > best.Fitness {
			best = cand
		}
	}
	return best
}

func RunTSPAlgorithm() TSPIndividual {
	rand.Seed(time.Now().UnixNano())
	initCities()

	// Инициализация популяции случайными маршрутами
	population := make([]TSPIndividual, TSPpopSize)
	for i := 0; i < TSPpopSize; i++ {
		route := rand.Perm(numCities)
		population[i] = TSPIndividual{Route: route}
		population[i].Evaluate()
	}

	var bestEver TSPIndividual
	bestEver.Fitness = -1.0

	// Эволюционный цикл
	for gen := 1; gen <= TSPmaxGenerations; gen++ {
		// Находим лучший маршрут в текущем поколении
		for i := range population {
			if population[i].Fitness > bestEver.Fitness {
				bestEver = population[i]
			}
		}

		// Создаем новое поколение
		newPop := make([]TSPIndividual, TSPpopSize)
		for i := 0; i < TSPpopSize; i++ {
			p1 := TSPTournamentSelection(population)
			p2 := TSPTournamentSelection(population)

			childRoute := OrderedCrossover(p1.Route, p2.Route)
			child := TSPIndividual{Route: childRoute}
			child.Mutate()
			child.Evaluate()
			newPop[i] = child
		}
		population = newPop
	}

	return bestEver
}
