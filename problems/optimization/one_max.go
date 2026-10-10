package optimization

import (
	"math/rand"
	"time"
)

const (
	targetLen            = 20   // Длина битовой строки
	OneMaxPopSize        = 100  // Размер популяции
	OneMaxMutationRate   = 0.05 // Вероятность мутации (5%)
	OneMaxMaxGenerations = 200  // Максимальное число поколений
)

// Индивидив представляет собой решение
type Individual struct {
	Genes   []int
	Fitness int
}

// Создание случайного индивидива
func generateRandomIndividual(length int) Individual {
	genes := make([]int, length)
	for i := 0; i < length; i++ {
		genes[i] = rand.Intn(2) // 0 или 1
	}
	return Individual{Genes: genes}
}

// Функция приспособленности: считает количество единиц
func (ind *Individual) Evaluate() {
	score := 0
	for _, gene := range ind.Genes {
		score += gene
	}
	ind.Fitness = score
}

// Кроссовер (одноточечный)
func Crossover(parent1, parent2 Individual) Individual {
	childGenes := make([]int, targetLen)
	// Случайная точка разрыва
	cp := rand.Intn(targetLen)

	for i := 0; i < targetLen; i++ {
		if i < cp {
			childGenes[i] = parent1.Genes[i]
		} else {
			childGenes[i] = parent2.Genes[i]
		}
	}
	return Individual{Genes: childGenes}
}

// Мутация
func (ind *Individual) Mutate() {
	for i := 0; i < targetLen; i++ {
		if rand.Float64() < OneMaxMutationRate {
			// Инвертируем бит
			if ind.Genes[i] == 0 {
				ind.Genes[i] = 1
			} else {
				ind.Genes[i] = 0
			}
		}
	}
}

// Турнирный отбор
func TournamentSelection(population []Individual) Individual {
	best := population[rand.Intn(len(population))]
	for i := 0; i < 2; i++ { // выбираем случайного конкурента
		competitor := population[rand.Intn(len(population))]
		if competitor.Fitness > best.Fitness {
			best = competitor
		}
	}
	return best
}

func RunGeneticAlgorithm() Individual {
	rand.Seed(time.Now().UnixNano())

	// Инициализация начальной популяции
	population := make([]Individual, OneMaxPopSize)
	for i := 0; i < OneMaxPopSize; i++ {
		population[i] = generateRandomIndividual(targetLen)
		population[i].Evaluate()
	}

	var bestEver Individual

	// Эволюционный цикл
	for gen := 1; gen <= OneMaxMaxGenerations; gen++ {
		// Находим лучшего в текущем поколении
		for i := 0; i < OneMaxPopSize; i++ {
			if population[i].Fitness > bestEver.Fitness {
				bestEver = population[i]
			}
		}

		// Если достигли идеала (все единицы), останавливаемся
		if bestEver.Fitness == targetLen {
			break
		}

		// Создаем новое поколение
		newPopulation := make([]Individual, OneMaxPopSize)
		for i := 0; i < OneMaxPopSize; i++ {
			parent1 := TournamentSelection(population)
			parent2 := TournamentSelection(population)

			child := Crossover(parent1, parent2)
			child.Mutate()
			child.Evaluate()

			newPopulation[i] = child
		}
		population = newPopulation
	}

	return bestEver
}
