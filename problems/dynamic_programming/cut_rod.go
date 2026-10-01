package dynamic_programming

func cutRod(p map[int]int, n int) int {
	r := make(map[int]int)
	for i := 0; i <= n; i++ {
		r[i] = -1
	}
	return memoizedCutRodAux(p, r, n)
}

func memoizedCutRodAux(p map[int]int, r map[int]int, n int) int {
	if r[n] >= 0 {
		return r[n]
	}
	var q int
	if n == 0 {
		q = 0
	} else if n != 0 {
		q = -1
		for i := 1; i <= n; i++ {
			q = max(q, p[i]+memoizedCutRodAux(p, r, n-i))
		}
	}
	r[n] = q

	return q
}
