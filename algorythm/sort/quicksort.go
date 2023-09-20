package sort

func Quicksort(a []int, l int, r int) {
	if l < r {
		q := partition(a, l, r)
		Quicksort(a, l, q-1)
		Quicksort(a, q+1, r)
	}
}

func partition(a []int, l int, r int) int {
	pivot := a[r]
	i := l - 1
	for j := l; j < r; j++ {
		if a[j] <= pivot {
			i++
			a[i], a[j] = a[j], a[i]
		}
	}
	a[i+1], a[r] = a[r], a[i+1]
	return i + 1
}
