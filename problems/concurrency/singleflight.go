package concurrency

import (
	"context"
	"sync"
)

// 100 горутин одновременно вызывают GetUser(42)
// В базу данных должен уйти ровно один запрос.
// Остальные должны дождаться результата первой горутины и получить тот же ответ.
// Cache Stampede (aka Thundering Herd, Dog-piling)
//
//  Сценарий. У тебя есть популярный ключ в кэше — homepage, user:42, top_products. По нему идёт, скажем, 5000 RPS. Ключ протухает по TTL.
//
//  В момент экспирации происходит следующее:
//
//  t=0.000  TTL истёк
//  t=0.001  Запрос A → cache miss → идёт в БД (займёт 200мс)
//  t=0.002  Запрос B → cache miss → тоже идёт в БД (не знает про A)
//  t=0.003  Запрос C → cache miss → тоже идёт в БД
//  ...
//  t=0.200  Все 1000 запросов, пришедших за эти 200мс, ушли в БД одновременно
//
//  Вместо 1 запроса в БД получаешь 1000 одновременных. БД захлёбывается, latency пробивает потолок, дальше по цепочке — таймауты в приложении, retry storm, всё каскадом ложится. Один
//  экспирившийся ключ уронил сервис.
//
//  Это и называется cache stampede — «стадо ломанулось в БД».
//
//  Классические механики защиты
//
//  1. Singleflight / request coalescing.
//  Из миссов только один идёт в БД, остальные ждут его результата. Решает 99% случаев в пределах процесса.
//
//  2. Stale-While-Revalidate (SWR).
//  Не удалять протухшее значение сразу — отдавать «старое», а параллельно в фоне обновлять. Пользователи не ждут ни разу. Есть в HTTP (Cache-Control: stale-while-revalidate=60), в nginx
//  (proxy_cache_use_stale updating), в Varnish (grace mode). В коде — сам ставишь soft_ttl < hard_ttl.
//
//  3. Probabilistic early expiration (XFetch).
//  Каждый запрос с вероятностью p, растущей по мере приближения TTL, «сам решает», что пора обновлять — до реального истечения. Один-два запроса обновят кэш заранее, остальные попадут уже в
//  свежее значение. Классическая статья: Vattani, Chierichetti, Lowenstein «Optimal Probabilistic Cache Stampede Prevention».
//
//  4. TTL с jitter'ом.
//  Если 1000 ключей поставлены с TTL=300s в одну секунду — они все протухнут в одну секунду через 5 минут. Добавляй случайный разброс: TTL = 300s ± 30s. Спасает от синхронного stampede по
//  разным ключам.
//
//  5. Distributed lock (для кросс-инстансного stampede).
//  Singleflight работает только в пределах процесса. Если у тебя 20 подов, каждый пойдёт в БД. Решение — блокировка в Redis: первый берёт lock, остальные ждут / читают stale. SET key:lock 1
//  NX EX 10.

type User struct {
	ID int
}

type SingleFlight struct {
	inFlight map[int]*Call
	m        sync.Mutex
}

type Call struct {
	done chan struct{}
	err  error
	user *User
}

func (s *SingleFlight) SingleFlight(ctx context.Context, key int, f func(int) (*User, error)) (*User, error) {
	if s.inFlight == nil {
		s.inFlight = make(map[int]*Call)
	}
	s.m.Lock()
	if call, ok := s.inFlight[key]; ok {
		s.m.Unlock()
		select {
		// если мапа не пустая, это значит, что первый запрос уже пошел выполнять f()
		// и мы ждем его завершения
		case <-call.done:
			return call.user, call.err
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	call := &Call{done: make(chan struct{})}

	s.inFlight[key] = call
	s.m.Unlock()

	// без лока т.к f может выполняться долго
	call.user, call.err = f(key)

	s.m.Lock()
	// удаляем, так как следующий запрос уже будет не inFlight
	delete(s.inFlight, key)
	s.m.Unlock()

	defer close(call.done)

	return call.user, call.err
}
