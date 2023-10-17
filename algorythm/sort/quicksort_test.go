package sort

import (
	"math"
	"math/rand"
	"reflect"
	"testing"
	"time"
)

func init() {
	rand.Seed(time.Now().Unix())
}

var quickTests = []struct {
	n        []int
	expected []int
}{
	{
		[]int{4, 1, 3, 2, 16, 9, 10, 14, 8, 7},
		[]int{1, 2, 3, 4, 7, 8, 9, 10, 14, 16},
	},
	{
		rand.Perm(10),
		makeRange(0, 9),
	},
	{
		rand.Perm(100),
		makeRange(0, 99),
	},
	{
		rand.Perm(int(math.Pow(2, 10))),
		makeRange(0, int(math.Pow(2, 10))-1),
	},
	{
		rand.Perm(int(math.Pow(2, 16))),
		makeRange(0, int(math.Pow(2, 16))-1),
	},
	{
		rand.Perm(int(math.Pow(2, 24))), // 16 777 216
		makeRange(0, int(math.Pow(2, 24))-1),
	},
	// 2 ^ 28 takes about 30 seconds with Quicksort(q.n, 0, len(q.n)-1)
	// 2 ^ 28 takes about 50 seconds with sort.Sort(sort.IntSlice(q.n))
	// during 2 ^ 32 test we get error: out of memory
}

func TestQuicksort(t *testing.T) {
	rand.Seed(time.Now().Unix())
	for _, q := range quickTests {
		startOrder := make([]int, len(q.n))
		copy(startOrder, q.n)
		Quicksort(q.n, 0, len(q.n)-1)
		if !reflect.DeepEqual(q.n, q.expected) {
			t.Errorf("Quicksort(%v): expected %v, actual %v ", startOrder, q.expected, q.n)
		}
	}
}

func makeRange(min, max int) []int {
	a := make(heapSliceInt, max-min+1)
	for i := range a {
		a[i] = min + i
	}
	return a
}
