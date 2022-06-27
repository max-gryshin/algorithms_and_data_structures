package main

import "fmt"

// todo: make it faster
func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
	l := len(nums1) + len(nums2)
	var startGreat, startLess []int
	var lenLess, lenGreat int
	hasEmpty := false
	if len(nums1) == 0 {
		startLess = nums2
		hasEmpty = true
	}
	if len(nums2) == 0 {
		startLess = nums1
		hasEmpty = true
	}
	if !hasEmpty {
		if nums1[0] >= nums2[0] {
			lenGreat = len(nums1)
			startGreat = make([]int, lenGreat)
			copy(startGreat, nums1)
			lenLess = len(nums2)
			startLess = make([]int, lenLess)
			copy(startLess, nums2)
		} else {
			lenGreat = len(nums2)
			startGreat = make([]int, lenGreat)
			copy(startGreat, nums2)
			lenLess = len(nums1)
			startLess = make([]int, lenLess)
			copy(startLess, nums1)
		}
		for i := 0; i < l; i++ {
			if startLess[i] >= startGreat[0] {
				startLess = append(startLess[:i], append([]int{startGreat[0]}, startLess[i:]...)...)
				startGreat = append(startGreat[:0], startGreat[1:]...)
			}
			if len(startGreat) == 0 {
				break
			}
			if len(startLess)-1 == i {
				startLess = append(startLess, startGreat...)
				break
			}
		}
	}
	if len(startLess)%2 == 0 {
		n := len(startLess) / 2
		return float64(startLess[n-1]+startLess[n]) / float64(2)
	} else {
		n := (len(startLess) - 1) / 2
		return float64(startLess[n])
	}
}

type SortedArrays struct {
	n1 []int
	n2 []int
}

func main() {
	sortedArrays := []SortedArrays{
		{
			[]int{1, 2},
			[]int{-1, 3},
		},
		{
			[]int{1},
			[]int{},
		},
		{
			[]int{1, 3},
			[]int{2},
		},
		{
			[]int{1, 2, 3},
			[]int{4, 5, 6},
		},
		{
			[]int{1, 3},
			[]int{2, 7},
		},
	}
	for _, arr := range sortedArrays {
		fmt.Println(findMedianSortedArrays(arr.n1, arr.n2))
	}
}
