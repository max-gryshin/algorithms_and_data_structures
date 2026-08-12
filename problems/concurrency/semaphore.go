package concurrency

import (
	"fmt"
)

// Что не хватает:
//
//  1. Наблюдаемость дропа. Классическая реализация drop-if-full сигнализирует о load-shedding: возвращает bool/error либо инкрементит метрику/лог. Сейчас — тихий дроп в default: return. В
//  проде саммирование пропущенных задач невозможно, что подрывает саму идею шедлоадинга (обычно ты хочешь видеть, что система в перегрузе).
//  2. Нет context/cancellation. Задача, попавшая в семафор, невыпихиваема — нельзя отменить долгую fn при shutdown. Каноничный вариант принимает ctx.
//  3. Приоритетность/fairness. Чистый select/default не гарантирует справедливость — если один producer быстрее других, он захватит все слоты. Для drop-if-full это обычно приемлемо, но стоит
//  упомянуть.
//  4. Идемпотентность освобождения не защищена. <-

// GoWithRecoverAndSemaphore executes fn in a goroutine, guarded by a counting semaphore (buffered channel sem).
//
// Concurrency pattern: non-blocking try-acquire with drop-if-full (a.k.a. load-shedding / best-effort dispatch).
// If the semaphore is saturated (cap(sem) tasks already in flight), the call is silently dropped and fn is NOT executed.
func GoWithRecoverAndSemaphore(fn func(), handlePanic func(err error), sem chan struct{}) {
	select {
	case sem <- struct{}{}:
		go func() {
			defer func() {
				<-sem
				if r := recover(); r != nil {
					if err, ok := r.(error); ok {
						handlePanic(err)
					} else {
						handlePanic(fmt.Errorf("panic occurred: %v", r))
					}
				}
			}()
			fn()
		}()
	default:
		return
	}
}
