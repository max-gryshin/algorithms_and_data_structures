# Worker Pool — вопросы для интервью

Разбор `problems/worker_pool.go`. Fan-out / fan-in паттерн: N workers обрабатывают задачи из `jobsCh`, M readers потребляют результаты из `chRes`. Синхронизация — двумя `sync.WaitGroup`.

---

## Корректность

### 1. Что произойдёт если `workers = 1`? А `workers = 0`?

**`workers = 1`** — `workers/2 == 0`. Буфер `chRes` нулевой, readers'ов **ноль**. Единственный worker пишет в `chRes`, читать некому — deadlock. `wg.Wait()` не вернётся никогда.

**`workers = 0`** — цикл создания workers не запустится, `jobsChan` заблокируется на первой отправке в небуферизованный канал (получателей нет), но `wg.Wait()` сразу вернёт (WaitGroup пустой), функция вернёт `nil`, а `jobsChan`-горутина повиснет навсегда — **утечка**.

Фикс:
```go
readers := workers / 2
if readers < 1 {
    readers = 1
}
```

Или явные параметры `producers, consumers int`.

### 2. Что если `jobs = nil` или пустой срез?

`jobsChan` сразу выполнит `defer close(ch)` без итераций → все workers выйдут из `for range` → `wg.Done()` → `wg.Wait()` вернётся → `close(chRes)` → readers выйдут из своих `range`. Всё корректно, функция вернёт `nil`.

### 3. Что если `Process` паникует?

Паника в любой горутине уронит весь процесс. Нужен `defer recover()` в worker'е с логированием или отправкой в error channel:

```go
go func(ch <-chan int, chRes chan<- int) {
    defer wg.Done()
    defer func() {
        if r := recover(); r != nil {
            errCh <- fmt.Errorf("worker panic: %v", r)
        }
    }()
    for v := range ch {
        chRes <- Process(v)
    }
}(jobsCh, chRes)
```

### 4. Что если `Process` возвращает ошибку?

Сменить контракт на `func Process(int) (int, error)` и отправлять `struct{Value int; Err error}` в `chRes`, либо завести отдельный `errCh`. Ошибка одной задачи **не должна** валить весь пул — читатель принимает решение (retry, abort, log).

### 5. Как доказать отсутствие утечки горутин?

- В тестах — библиотека `go.uber.org/goleak`:
```go
func TestRunWorkers(t *testing.T) {
    defer goleak.VerifyNone(t)
    RunWorkers([]int{1, 2, 3}, 4)
}
```
- Инвариант: каждая горутина завершается по закрытию своего входного канала. `jobsChan` закрывает `jobsCh`, основная функция закрывает `chRes`. Значит все производные горутины детерминированно завершатся.

---

## Контракт API

### 6. Функция возвращает `nil`. Каким должен быть контракт?

Три реалистичных варианта:
1. **`func RunWorkers(jobs []int, workers int) []int`** — вернуть результаты. Собирать в срез внутри reader'ов (под мьютексом) или через отдельный агрегатор-горутину.
2. **`func RunWorkers(jobs []int, workers int) error`** — обработка ради побочного эффекта (запись в БД). Возвращать первую ошибку или объединённую.
3. **`func RunWorkers(ctx context.Context, jobs []int, workers int) (<-chan Result, error)`** — стримовый API. Клиент читает результаты по мере готовности.

Возвращаемое `[]int` без сбора — контрактная ошибка.

### 7. Как гарантировать порядок результатов?

Передавать индекс через канал:
```go
type Task struct{ Idx, Val int }
type Result struct{ Idx, Val int }
```
Reader раскладывает результаты в срез по `Idx`. Порядок восстанавливается на этапе агрегации.

### 8. Как реализовать отмену?

Через `context.Context`:
```go
func RunWorkers(ctx context.Context, jobs []int, workers int) error {
    // workers:
    for {
        select {
        case <-ctx.Done():
            return
        case v, ok := <-jobsCh:
            if !ok { return }
            select {
            case chRes <- Process(v):
            case <-ctx.Done():
                return
            }
        }
    }
}
```

Оба `select` нужны: и на чтение из `jobsCh`, и на запись в `chRes` — иначе можно застрять на записи в незанятый канал.

---

## Производительность и тюнинг

### 9. Почему буфер именно `workers/2`? Как выбирать эмпирически?

`workers/2` — эвристика без обоснования. Правильный подход:
- Профилировать: `go test -bench -benchmem`, посмотреть на `pprof` где времена ожидания на каналах.
- Правило Литтла: `буфер ≈ (average_send_rate) × (average_read_latency)`. Если producer быстрее consumer'а и всплески короткие — буфер поглощает пики.
- Если consumer стабильно медленнее — буфер размером N просто отсрочит блокировку producers на N сообщений, скорость не изменится (упрёшься в bottleneck).

### 10. Где сейчас bottleneck?

Сейчас `Process` и `reader` оба `time.Sleep(50ms)`. Reader'ов `workers/2`, workers'ов `workers`. Пропускная способность:
- workers производят `workers / 50ms` = W/50 результатов/мс
- readers потребляют `(workers/2) / 50ms` = W/100 результатов/мс

**Bottleneck — readers** (в 2 раза медленнее). Увеличение до `workers` readers'ов уравняет производительность.

### 11. CPU-bound vs IO-bound `Process`?

- **CPU-bound**: оптимум `workers ≈ runtime.NumCPU()`. Больше — просто увеличит переключения контекста.
- **IO-bound** (сетевой запрос, ожидание диска): оптимум сильно больше — 100, 1000, зависит от ресурсов внешней системы. Ограничитель — memory footprint и лимиты БД/API.

### 12. Что если задач миллион?

Сейчас `jobs []int` весь в памяти — не подойдёт. Переделать источник задач в стрим:
```go
func RunWorkers(ctx context.Context, jobsCh <-chan int, workers int) error
```
Тот, кто вызывает, отдаёт готовый канал (читает из БД, файла, сети). `jobsChan` из твоего кода — отдельная утилита адаптер.

---

## Устойчивость

### 13. Как ограничить время выполнения одного `Process`?

`context.WithTimeout` + `Process(ctx, v)`. Реализация `Process` должна периодически проверять `ctx.Done()`. Если нельзя менять — запустить `Process` в отдельной горутине с select по таймеру, но это утечка горутины при зависании (её не остановить снаружи в Go).

### 14. Как повторять упавшие задачи?

- Простой ретрай внутри worker'а: `for attempt := 0; attempt < 3; attempt++`.
- С экспоненциальной задержкой между попытками.
- Если retry на уровне пула — вернуть задачу обратно в `jobsCh` (но осторожно с бесконечным циклом на "плохих" задачах — нужен счётчик попыток в `Task`).

### 15. Graceful shutdown?

Разделить два сигнала:
- **stop accepting new**: закрыть `jobsCh` из внешнего кода. Workers доработают текущий буфер и выйдут.
- **abort immediately**: `ctx.Cancel()`. Workers прерывают текущую задачу через ctx.

Классический паттерн — оба сразу: сначала пробуем graceful, через N секунд ctx.Cancel().

---

## Архитектура

### 16. Где здесь fan-out, а где fan-in?

- **Fan-out**: одна точка (jobsCh) → N workers параллельно берут задачи.
- **Fan-in**: N workers пишут в один канал `chRes` → M readers его разбирают.

Здесь **двойной fan-out**: jobs → workers (fan-out), workers → chRes (fan-in), chRes → readers (fan-out).

### 17. Почему readers должны быть в отдельном WaitGroup?

Если использовать `wg` для обоих:
1. Workers пишут в `chRes`, ждут пока reader примет.
2. Readers сидят в `for range chRes`, ждут `close(chRes)`.
3. Основная горутина ждёт `wg.Wait()`.
4. `close(chRes)` вызывается **после** `wg.Wait()`.

Круг: `wg.Wait()` ждёт readers → readers ждут `close` → `close` после `wg.Wait()` → **deadlock**.

Разделение WaitGroups позволяет закрыть канал в правильный момент — **после** окончания producers, но **до** окончания consumers.

### 18. Можно ли обойтись без `close(chRes)`?

Нет. Readers сидят в `for range chRes` — без close они никогда не выйдут. Плюс `wgRes.Wait()` заблокируется навсегда. `close` — единственный сигнал "больше данных не будет".

Альтернатива — не использовать `range`, а `select` с `ctx.Done()`. Тогда close не обязателен, но нужен явный контекст.

### 19. Почему `jobsCh` небуферизованный, а `chRes` — буферизованный?

Осознанной причины нет — обычно это выбирается симметрично. `jobsCh` небуферизованный даёт **back-pressure**: producer в `jobsChan` не сможет положить в канал больше, чем workers успевают забирать → память под очередь не растёт. Это хорошо когда `jobs` большой.

`chRes` буферизованный сглаживает всплески между workers и readers. Если workers одновременно закончат `Process` — они не блокируются на send синхронно, а раскладывают результаты в буфер.

Обратный вариант (буферизованный `jobsCh`, небуферизованный `chRes`) — валиден, но теряет back-pressure на входе.

---

## Тестирование

### 20. Как это протестировать?

Проблемы текущего кода для тестов:
- Побочный эффект `fmt.Println` вместо возврата — нельзя проверить результат.
- `time.Sleep(50ms)` — тест медленный.

Рефакторинг для тестируемости:
```go
type Config struct {
    Workers int
    Readers int
    BufSize int
    Process func(int) int
    Emit    func(int)
}

func RunWorkers(ctx context.Context, jobs []int, cfg Config) error
```

Тесты:
1. **Функциональный**: собрать в срез через `Emit`, проверить `len` и множество (порядок не гарантирован).
2. **Утечки**: `goleak.VerifyNone(t)` в defer.
3. **Race**: `go test -race`.
4. **Отмена**: передать `ctx` с cancel, убедиться что функция вернётся.
5. **Паника**: передать `Process`, который паникует на конкретном входе — убедиться что остальные задачи всё равно обработаны.

---

## Что хочу увидеть от кандидата

Средний кандидат:
- Знает, почему нужен `defer wg.Done()`.
- Может рассказать что делает `range` по каналу.

Хороший:
- Ловит deadlock при `workers=1`.
- Замечает `return nil` в функции с сигнатурой `[]int`.
- Понимает роль buferized vs unbuffered.

Сильный:
- Сходу говорит про два WaitGroup и объясняет **почему** нельзя один.
- Спрашивает про `context.Context` до того как ты сам предложил.
- Замечает отсутствие обработки паники и ошибок.
- Обсуждает back-pressure и pprof для тюнинга.
