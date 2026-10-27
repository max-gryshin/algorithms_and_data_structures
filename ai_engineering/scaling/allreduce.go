package scaling

import (
	"sync"
	"sync/atomic"
)

// RingAllReduce sums the vectors of n workers element-wise so that every worker ends up
// with the full sum — the operation data-parallel training runs on gradients after every step.
//
// Workers are goroutines connected in a ring by channels (worker i sends only to i+1).
// The vector is cut into n chunks and the algorithm runs in two phases of n−1 steps:
//
//	reduce-scatter: each step every worker passes one partially summed chunk forward and adds
//	                the chunk it receives; afterwards worker i owns the complete sum of chunk i+1;
//	all-gather:     the complete chunks travel around the ring once more and overwrite the stale copies.
//
// Every worker sends 2·(n−1) chunks of S/n values, i.e. 2·(n−1)/n·S values in total —
// almost independent of n. The returned bytesPerWorker counts float32 values sent by worker 0.
func RingAllReduce(inputs [][]float32) (outputs [][]float32, bytesPerWorker int64) {
	n := len(inputs)
	size := len(inputs[0])
	bufs := make([][]float32, n)
	for i := range inputs {
		bufs[i] = append([]float32(nil), inputs[i]...)
	}
	if n == 1 {
		return bufs, 0
	}
	chunk := func(c int) (lo, hi int) {
		c = ((c % n) + n) % n
		return c * size / n, (c + 1) * size / n
	}

	// links[i] carries messages from worker i to worker (i+1) mod n.
	links := make([]chan []float32, n)
	for i := range links {
		links[i] = make(chan []float32, 1)
	}
	var sent atomic.Int64

	var wg sync.WaitGroup
	for i := range n {
		wg.Go(func() {
			buf := bufs[i]
			out, in := links[i], links[(i-1+n)%n]
			send := func(c int) {
				lo, hi := chunk(c)
				msg := append([]float32(nil), buf[lo:hi]...) // copy: the receiver must not alias our buffer
				if i == 0 {
					sent.Add(int64(len(msg)) * 4)
				}
				out <- msg
			}

			// Reduce-scatter: at step s send chunk i−s, receive chunk i−s−1 and add it.
			for s := range n - 1 {
				send(i - s)
				msg := <-in
				lo, _ := chunk(i - s - 1)
				for k, v := range msg {
					buf[lo+k] += v
				}
			}
			// All-gather: at step s send chunk i+1−s, receive chunk i−s and overwrite it.
			for s := range n - 1 {
				send(i + 1 - s)
				msg := <-in
				lo, _ := chunk(i - s)
				copy(buf[lo:], msg)
			}
		})
	}
	wg.Wait()
	return bufs, sent.Load()
}

// NaiveAllReduceRootBytes is what the root receives and sends in the naive scheme
// (everyone sends the full vector to worker 0, which broadcasts the sum back): 2·(n−1)·S values.
func NaiveAllReduceRootBytes(n, size int) int64 { return 2 * int64(n-1) * int64(size) * 4 }
