package problems

import "testing"

var sortedSlices = []struct {
	a        []int
	b        []int
	expected []int
}{
	{
		[]int{1, 3, 5, 7, 9},
		[]int{2, 4, 6, 8, 10},
		[]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
	},
	{
		[]int{1, 3, 5, 7, 9, 13, 17, 21, 31, 55, 77},
		[]int{2, 4, 6, 8, 10},
		[]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 17, 21, 31, 55, 77},
	},
	{
		[]int{1, 3, 5, 7, 9},
		[]int{2, 4, 6, 8, 10, 13, 17, 21, 31, 55, 77},
		[]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 17, 21, 31, 55, 77},
	},
}

func TestMergeSort(t *testing.T) {
	for _, v := range sortedSlices {
		res := mergeSort(v.a, v.b)
		if len(res) != len(v.expected) {
			t.Errorf(
				"The len of slice mergeSort(%v, %v) not correct: expected %v, actual %v",
				v.a,
				v.b,
				v.expected,
				res,
			)
		}
		for i := 0; i < len(res); i++ {
			if v.expected[i] != res[i] {
				t.Errorf(
					"The values of mergeSort(%v, %v) not correct: expected %v, actual %v",
					v.a,
					v.b,
					v.expected[i],
					res[i],
				)
			}
		}
	}
}
