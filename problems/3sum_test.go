package problems

import "testing"

var threeSumData = []struct {
	n        []int
	expected [][]int
}{
	{
		[]int{0, 3, 0, 1, 1, -1, -5, -5, 3, -3, -3, 0},
		[][]int{{-3, 0, 3}, {-1, 0, 1}, {0, 0, 0}},
	},
	{
		[]int{-1, 0, 1, 2, -1, -4},
		[][]int{{-1, -1, 2}, {-1, 0, 1}},
	},
	{
		[]int{-1, 0, 1, 2, -1, -4, -2, -3, 3, 0, 4},
		[][]int{{-4, 0, 4}, {-4, 1, 3}, {-3, -1, 4}, {-3, 0, 3}, {-3, 1, 2}, {-2, -1, 3}, {-2, 0, 2}, {-1, -1, 2}, {-1, 0, 1}},
	},
	{
		[]int{-2, 0, 1, 1, 2},
		[][]int{{-2, 0, 2}, {-2, 1, 1}},
	},
	{
		[]int{0, 0, 0},
		[][]int{{0, 0, 0}},
	},
	{
		[]int{0, 1, 1},
		nil,
	},
}

func TestThreeSum(t *testing.T) {
	for _, v := range threeSumData {
		res := threeSum(v.n)
		if res == nil && v.expected != nil {
			t.Errorf(
				" the result of TestThreeSum(%v) not correct: expected %v, actual %v",
				v.n,
				v.expected,
				res,
			)
		}
		for i := 0; i < len(res); i++ {
			for j := 0; j < len(res[i]); j++ {
				if res[i][j] != v.expected[i][j] {
					t.Errorf(
						" the result of TestThreeSum(%v) not correct: expected %v, actual %v",
						v.n,
						v.expected,
						res,
					)
				}
			}
		}
	}
}
