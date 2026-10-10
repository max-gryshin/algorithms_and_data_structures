package optimization

import (
	"math"
	"math/rand"
	"time"
)

// Набор гиперпараметров, которые настраивает ГА
type Hyperparameters struct {
	MaxDepth     int     // Глубина (целое число от 2 до 10)
	LearningRate float64 // Скорость обучения (вещественное от 0.01 до 0.3)
	Threshold    float64 // Порог классификации (вещественное от 0.4 до 0.9)
	Fitness      float64 // Качество модели (симулируем тестовой функцией)
}

const (
	popSizeML      = 50
	mutationRateML = 0.15
	maxGensML      = 100
)

// Симуляция обучения модели с заданными гиперпараметрами
// В реальности здесь вызывалась бы библиотека ML, которая возвращает точность на валидации
func evaluateMLModel(h *Hyperparameters) float64 {
	// Имитируем целевую функцию: допустим, идеальные параметры где-то посередине диапазона,
	// а ГА должен их "нащупать".
	depthScore := 1.0 - math.Abs(float64(h.MaxDepth-6))/6.0
	lrScore := 1.0 - math.Abs(h.LearningRate-0.15)/0.15
	thScore := 1.0 - math.Abs(h.Threshold-0.75)/0.35

	// Комплексная метрика (например, F1-score бизнеса)
	score := (depthScore*0.3 + lrScore*0.3 + thScore*0.4) * 100
	if score < 0 {
		return 0
	}
	return score
}

func randomHP() Hyperparameters {
	return Hyperparameters{
		MaxDepth:     rand.Intn(9) + 2,           // 2..10
		LearningRate: rand.Float64()*0.29 + 0.01, // 0.01..0.3
		Threshold:    rand.Float64()*0.5 + 0.4,   // 0.4..0.9
	}
}

// Кроссовер (арифметический / смешивание)
func HPCrossover(p1, p2 Hyperparameters) Hyperparameters {
	child := Hyperparameters{
		MaxDepth:     p1.MaxDepth,
		LearningRate: p1.LearningRate,
		Threshold:    p1.Threshold,
	}

	if rand.Float64() < 0.5 {
		child.MaxDepth = p1.MaxDepth
	} else {
		child.MaxDepth = p2.MaxDepth
	}

	if rand.Float64() < 0.5 {
		child.LearningRate = p1.LearningRate
	} else {
		child.LearningRate = p2.LearningRate
	}

	if rand.Float64() < 0.5 {
		child.Threshold = p1.Threshold
	} else {
		child.Threshold = p2.Threshold
	}

	return child
}

// Мутация с учетом типов данных
func (h *Hyperparameters) HPMutate() {
	if rand.Float64() < mutationRateML {
		h.MaxDepth += rand.Intn(3) - 1 // сдвиг на -1, 0 или +1
		if h.MaxDepth < 2 {
			h.MaxDepth = 2
		}
		if h.MaxDepth > 10 {
			h.MaxDepth = 10
		}
	}

	if rand.Float64() < mutationRateML {
		h.LearningRate += (rand.Float64() - 0.5) * 0.05
		if h.LearningRate < 0.01 {
			h.LearningRate = 0.01
		}
		if h.LearningRate > 0.3 {
			h.LearningRate = 0.3
		}
	}

	if rand.Float64() < mutationRateML {
		h.Threshold += (rand.Float64() - 0.5) * 0.05
		if h.Threshold < 0.4 {
			h.Threshold = 0.4
		}
		if h.Threshold > 0.9 {
			h.Threshold = 0.9
		}
	}
}

func RunAutoMLAlgorithm() Hyperparameters {
	rand.Seed(time.Now().UnixNano())

	// Инициализация начальной популяции случайными гиперпараметрами
	population := make([]Hyperparameters, popSizeML)
	for i := range population {
		population[i] = randomHP()
		population[i].Fitness = evaluateMLModel(&population[i])
	}

	var bestHP Hyperparameters
	bestHP.Fitness = -1.0

	// Эволюционный цикл для поиска оптимальных гиперпараметров
	for gen := 1; gen <= maxGensML; gen++ {
		// Находим лучшего кандидата в текущем поколении
		for _, hp := range population {
			if hp.Fitness > bestHP.Fitness {
				bestHP = hp
			}
		}

		// Создаем новое поколение через кроссовер и мутацию
		newPop := make([]Hyperparameters, popSizeML)
		for i := 0; i < popSizeML; i++ {
			// Простой турнирный отбор
			p1 := population[rand.Intn(popSizeML)]
			p2 := population[rand.Intn(popSizeML)]
			parent := p1
			if p2.Fitness > p1.Fitness {
				parent = p2
			}

			child := HPCrossover(parent, p2)
			child.HPMutate()
			child.Fitness = evaluateMLModel(&child)
			newPop[i] = child
		}
		population = newPop
	}

	return bestHP
}
