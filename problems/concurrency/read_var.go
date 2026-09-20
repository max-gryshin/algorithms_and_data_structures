package concurrency

import (
	"strconv"
	"sync"
)

func BrokeProc() {
	x := 0
	y := 0

	r1 := 0
	r2 := 0

	f1 := func(wgg *sync.WaitGroup) {
		x = 1
		r1 = y
		wgg.Done()
	}
	f2 := func(wgg *sync.WaitGroup) {
		y = 1
		r2 = x
		wgg.Done()
	}
	const iter = 1000000
	wg := sync.WaitGroup{}
	for i := 0; i < iter; i++ {
		wg.Add(2)
		go f1(&wg)
		go f2(&wg)
		wg.Wait()
		if r1 == 0 && r2 == 0 {
			panic("we broke processor at " + strconv.Itoa(i) + " iteration")
		}
	}
}
