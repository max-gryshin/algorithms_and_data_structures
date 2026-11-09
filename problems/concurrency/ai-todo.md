1. sync.Cond — условные переменные

Суть. Примитив «жди, пока не выполнится условие + разбуди когда изменилось». Работает поверх Locker (обычно *sync.Mutex).

type Queue struct {
mu    sync.Mutex
cond  *sync.Cond
items []int
max   int
}

func NewQueue(max int) *Queue {
q := &Queue{max: max}
q.cond = sync.NewCond(&q.mu)
return q
}

func (q *Queue) Push(x int) {                                                                                                              
q.mu.Lock()
for len(q.items) == q.max {   // именно for, не if!
q.cond.Wait()             // атомарно: unlock + park; при разбуждении — lock
}
q.items = append(q.items, x)
q.cond.Signal()               // разбудить одного ждущего
q.mu.Unlock()
}

func (q *Queue) Pop() int {
q.mu.Lock()
for len(q.items) == 0 {
q.cond.Wait()
}
x := q.items[0]
q.items = q.items[1:]
q.cond.Signal()
q.mu.Unlock()
return x
}

Правила:
- Wait() вызывать только в цикле проверки условия — spurious wakeups + возможность, что за время wake кто-то уже забрал ресурс.
- Условие меняется под тем же mutex'ом, что и Signal/Broadcast.
- Signal — разбудить одного, Broadcast — всех ждущих.

Где реально используется:
- database/sql — пул соединений: горутина ждёт, пока освободится connection, вместо busy-wait.
- net/http — h2c/http2 в стандартной либе, координация потоков.
- Кастомные bounded queue / semaphore, где нужны разные условия просыпания (не хватает одного канала).

Почему в Go его не любят и предпочитают каналы.
Rob Pike прямо говорил: «don't use sync.Cond, use channels». В большинстве задач chan struct{} + select понятнее и не имеет ловушек с Wait() вне цикла. Cond нужен, когда:
1. Много разных условий на одном разделяемом состоянии (canal одна = одно условие).
2. Broadcast'ом нужно разбудить произвольное число ждущих одновременно (с каналом это close(ch), но канал одноразовый).
3. Горячий путь — Cond дешевле, чем создание/закрытие каналов на каждую итерацию.

  ---
2. sync.Pool — переиспользование объектов

Суть. Пул временных объектов, снижающий давление на GC. Каждый P (процессор) имеет локальный подпул → почти безлоковый доступ. GC периодически чистит пул.

var bufPool = sync.Pool{
New: func() any { return new(bytes.Buffer) },
}

func handler(w http.ResponseWriter, r *http.Request) {                                                                                     
buf := bufPool.Get().(*bytes.Buffer)
defer func() {
buf.Reset()          // важно очистить перед возвратом
bufPool.Put(buf)
}()
// работаем с buf...
}

Где реально используется:
- fmt (stdlib) — пул pp (printer state), поэтому fmt.Sprintf дешёвый в горячем пути.
- encoding/json — пул scanner state.
- net/http — пул bufio.Reader/Writer для соединений.
- gin, fiber, echo — пул Context-объектов на запрос.
- zap, zerolog — пул буферов и encoder'ов.
- protobuf — пул для marshaling.

Правила:
- Только для stateless или явно Reset()-нутых объектов. Если забыл почистить — предыдущие данные утекут в следующий запрос (реальный CVE-класс: leaking user data через reused buffer).
- Нельзя полагаться на выживание объекта в пуле. GC может выкинуть в любой момент, Get вернёт новый через New. То есть Pool — оптимизация, а не хранилище.
- Не для connection-подобных ресурсов. Для БД/HTTP-соединений — свой pool (database/sql, http.Transport), потому что там нужен lifecycle (открытие/закрытие/health check), которого у
  sync.Pool нет.
- Осторожно с большими объектами — sync.Pool может держать в памяти много одинаковых буферов. bytes.Buffer после append'а на 10MB останется в пуле с backing array 10MB. Решение — не класть
  обратно, если cap слишком велик.

Когда НЕ помогает. Если объекты «долгоживущие» (пережили несколько GC-циклов) или редко создаются — pool бесполезен. Мерять надо: go test -bench . -benchmem до и после.

  ---
3. Lock-free (atomics, CAS)

Суть. Синхронизация без блокировок — через sync/atomic (Load/Store/Add/CompareAndSwap) и compare-and-swap-циклы. С Go 1.19 появились типизированные атомики: atomic.Int64,
atomic.Pointer[T], atomic.Bool.

// Классический CAS-цикл: неблокирующее обновление
type Counter struct {
v atomic.Int64
}

func (c *Counter) Inc() { c.v.Add(1) }

// Кастомная логика через CAS
type MinTracker struct {                                                                                                                   
min atomic.Int64
}

func (m *MinTracker) Observe(x int64) {
for {
cur := m.min.Load()
if x >= cur {                                                                                                                      
return
}
if m.min.CompareAndSwap(cur, x) {
return
}
// проиграли гонку — читаем свежее значение и пробуем снова
}
}

atomic.Value / atomic.Pointer[T] — самый частый паттерн: copy-on-write конфиг.

var cfg atomic.Pointer[Config]

// hot path — читатели
func GetConfig() *Config { return cfg.Load() }

// холодный путь — обновление
func Reload(newCfg *Config) { cfg.Store(newCfg) }

Читатели вообще не берут lock. Идеально для конфигов, feature flags, routing tables — читается миллионы раз, обновляется раз в минуту.

Где реально используется:
- Метрики — prometheus/client_golang использует atomics в counter/gauge (сотни тысяч инкрементов в секунду).
- sync.Once внутри — атомарный флаг done.
- sync.Map — CAS-хвосты для lock-free read path.
- context.Context — атомик для флага «cancelled».
- Ring buffer'ы, lock-free очереди — LMAX Disruptor-подобные структуры.
- Configuration hot-reload — почти во всех сервисах через atomic.Pointer.

Ловушки:
- CAS-циклы под высокой contention становятся хуже мьютекса — CPU жжётся впустую на retry'ях. Мерять!
- Порядок операций (memory ordering). В Go все atomics — sequential consistency, но легко забыть: «атомарный флаг + не-атомарные данные» = гонка. Читать статью Russ Cox «Hardware Memory
  Models».
- Atomicity ≠ correctness. x.Add(1); if x.Load() == 10 {...} — это две отдельные атомарные операции, между ними всё что угодно. Атомарность не даёт транзакционности.
- Порядок полей. До Go 1.19 нужно было руками выравнивать 64-битные поля на 32-битных платформах. С типизированными atomic.Int64 — компилятор сам делает.

  ---
4. Signal pattern

Тут несколько разных вещей называют «signal pattern». Разберу все.

4a. Broadcast через close(chan)

Один канал = один общий сигнал многим ждущим.

type Broker struct {
ready chan struct{}
}

func NewBroker() *Broker {
return &Broker{ready: make(chan struct{})}
}

func (b *Broker) Wait() { <-b.ready }          // сколько угодно горутин ждут
func (b *Broker) Signal() { close(b.ready) }   // одним close будят всех

Это идиоматичный «broadcast» в Go — вместо sync.Cond.Broadcast(). Именно так работают context.Context.Done(), http.Server.Shutdown, все «готовность/отмена».

Ограничение: канал одноразовый. Повторный сигнал = новый канал.

4b. os/signal — обработка сигналов ОС для graceful shutdown

ctx, stop := signal.NotifyContext(context.Background(),
syscall.SIGINT, syscall.SIGTERM)                                                                                                       
defer stop()

srv := &http.Server{Addr: ":8080", Handler: mux}

go func() {
if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {                                                            
log.Fatal(err)
}
}()

<-ctx.Done() // ждём SIGINT/SIGTERM

shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()
if err := srv.Shutdown(shutdownCtx); err != nil {
log.Printf("graceful shutdown failed: %v", err)
}

signal.NotifyContext (Go 1.16+) — самый чистый способ. До этого — signal.Notify(ch, ...) + свой канал.

Каскад останова в проде обычно такой: SIGTERM → перестать принимать новые запросы (srv.Shutdown) → дождаться дренажа in-flight запросов → закрыть БД/Kafka-consumer'ы → выйти.

4c. Trigger/reset — семафор-подобная сигнализация

chan struct{} с буфером 1 и select на отправке:

type Trigger struct {
ch chan struct{}
}

func NewTrigger() *Trigger { return &Trigger{ch: make(chan struct{}, 1)} }

func (t *Trigger) Fire() {
select {
case t.ch <- struct{}{}: // положили сигнал
default:                 // уже есть — coalesce
}
}

func (t *Trigger) Wait() <-chan struct{} { return t.ch }

Классический «debounce» / «coalescing signal» — 100 инвалидаций между двумя wake-up'ами превратятся в один. Полезно для «пересчитай state, но не чаще раза в секунду».

  ---
5. errgroup — параллельные задачи с ошибкой и отменой

Библиотека: golang.org/x/sync/errgroup.

Что даёт:
1. Wait возвращает первую непустую ошибку.
2. С WithContext — при ошибке любой из горутин ctx отменяется → остальные могут завершиться раньше.
3. SetLimit(n) (Go 1.20+) — ограничение параллелизма.

func FetchAll(ctx context.Context, ids []int) ([]User, error) {
g, ctx := errgroup.WithContext(ctx)
g.SetLimit(10) // не больше 10 параллельных

      users := make([]User, len(ids))
      for i, id := range ids {
          i, id := i, id                                                                                                                     
          g.Go(func() error {
              u, err := fetchUser(ctx, id)
              if err != nil {
                  return err // отменит ctx → отменит остальные fetchUser
              }
              users[i] = u
              return nil
          })
      }
      if err := g.Wait(); err != nil {
          return nil, err
      }
      return users, nil
}

Где реально используется:
- Fan-out к нескольким downstream-сервисам (BFF, aggregator-паттерн).
- Параллельный batch-fetch из БД/кэша.
- Kubernetes controllers — параллельные reconcile.
- Docker / containerd — параллельная сборка слоёв.

Ловушки:
- g.Go(func() error { ... }) — closure над loop-переменной. До Go 1.22 нужен был i := i перед Go. С 1.22 переменные в for per-iteration — уже нет.
- Первая ошибка «выигрывает». Остальные ошибки от других горутин теряются. Если нужны все — собирай сам через канал/срез под мьютексом.
- SetLimit вызывать до первого g.Go.
- errgroup не запустит новые горутины после отмены ctx — но уже запущенные должны сами проверить ctx.Err().

  ---
6. Pipeline

Суть. Data flow как цепочка стадий, соединённых каналами. Каждая стадия — горутина или пул горутин.

// stage 1: генератор
func gen(ctx context.Context, nums ...int) <-chan int {
out := make(chan int)
go func() {
defer close(out)
for _, n := range nums {                                                                                                           
select {
case out <- n:
case <-ctx.Done():
return
}
}
}()
return out
}

// stage 2: трансформация
func square(ctx context.Context, in <-chan int) <-chan int {
out := make(chan int)
go func() {
defer close(out)
for n := range in {
select {
case out <- n * n:
case <-ctx.Done():
return
}
}
}()
return out
}

// использование
ctx, cancel := context.WithCancel(context.Background())
defer cancel()
for v := range square(ctx, gen(ctx, 1, 2, 3, 4)) {
fmt.Println(v)
}

Идиоматика:
- Каждая стадия сама закрывает свой output (через defer close(out)).
- Каждая стадия читает input до конца (в range) — иначе апстрим-горутины залипнут на out <- ....
- Для параллелизма внутри стадии — fan-out/fan-in: N горутин из одного канала пишут в один общий (через merge-функцию).
- Для остановки — общий ctx во всех стадиях + select на отправке.

Backpressure (по поводу твоего пункта из todo).
Небуферизированный канал в pipeline — автоматическая backpressure: продюсер блокируется на send, пока консьюмер не готов. Буферизованный канал (make(chan int, N)) — сглаживает спайки: если
продюсер выдал burst из N сообщений — они лягут в буфер, продюсер не заблокируется, консьюмер разгребёт в своём темпе. Но если burst > N устойчиво — буфер не поможет, всё равно упрётся.
Практика: буфер = 1×–2× пиковый burst, не больше — иначе прячешь проблемы производительности и раздуваешь память.

Где реально используется:
- ETL / стриминговая обработка — Kafka consumer → decode → filter → enrich → sink.
- io в stdlib — bufio.Scanner, io.Pipe, io.TeeReader — те же идеи, но синхронно.
- Ffmpeg-подобные обёртки, log processing, tail'ы — parse → enrich → ship.
- gRPC streaming handlers — часто внутри развёрнут pipeline.
- go-streams, benthos, conduit — целые фреймворки поверх этого паттерна.

Ловушки:
- Goroutine leak — стадия зависла на out <- ..., потому что consumer вышел без drain'а. Всегда либо for range до конца, либо ctx-cancel + select.
- Порядок vs параллелизм. Fan-out ломает порядок. Если он нужен — либо не fan-out'ить, либо восстанавливать порядок через sequence numbers.
- Ошибки в pipeline. Каналы не переносят ошибки нативно. Обычно передают struct { Val T; Err error } или используют errgroup, где каждая стадия — g.Go.

  ---
Что ещё стоит добавить в твой todo
  
---
Что ещё стоит добавить в твой todo

Твой список хорош, но чего в нём не хватает для Senior-уровня:

- context.Context deep dive — deadline propagation, WithCancelCause (1.20+), context.AfterFunc (1.21+), почему WithValue — антипаттерн для не-request-scoped данных.
- sync.Once и его новые варианты 1.21+ — OnceFunc, OnceValue, OnceValues.
- Fan-out / Fan-in — у тебя есть в todo, но отдельным примером стоит: merge-функция с WaitGroup + close(out) в defer.
- Bounded parallelism — почти дубль semaphore, но полезно как отдельный паттерн (chan struct{} буфер N).
- Actor model / mailbox — одна горутина владеет состоянием, все обращения к ней — через input-канал. Альтернатива мьютексам, часто чище для сложного state machine.
- Barrier / Latch — как в Java CountDownLatch. В Go делается через sync.WaitGroup (для «сколько-то раз») или chan struct{} (для one-shot).
- Debounce / Throttle — обёртки над time.Timer / time.Ticker. Часто путают, стоит развести.
- Circuit breaker — sony/gobreaker, hystrix-go. Concurrent-safe state machine (closed/open/half-open).
- Bulkhead — изоляция ресурсов, чтобы один медленный downstream не съел все горутины/коннекты. По сути — семафор per-tenant/per-service.