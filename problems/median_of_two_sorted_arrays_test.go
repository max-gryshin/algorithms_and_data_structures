package problems

import "testing"

var sortedArrays = []struct {
	n1       []int
	n2       []int
	expected float64
}{
	{
		[]int{1, 2},
		[]int{-1, 3},
		1.5,
	},
	{
		[]int{1},
		[]int{},
		1,
	},
	{
		[]int{1, 3},
		[]int{2},
		2,
	},
	{
		[]int{1, 2, 3},
		[]int{4, 5, 6},
		3.5,
	},
	{
		[]int{1, 3},
		[]int{2, 7},
		2.5,
	},
}

func TestFindMedianSortedArrays(t *testing.T) {
	for _, v := range sortedArrays {
		res := findMedianSortedArrays(v.n1, v.n2)
		if v.expected != res {
			t.Errorf(
				" the result of findMedianSortedArrays(%v, %v) not correct: expected %d, actual %d",
				v.n1,
				v.n2,
				uint64(v.expected),
				uint64(res),
			)
		}
	}
}
