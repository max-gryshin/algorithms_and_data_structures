package dynamic_programming

import "math"

type table struct {
	i int
	j int
}

type mTable map[table]int

func (m mTable) Get(i, j int) int {
	return m[table{i: i, j: j}]
}

func (m mTable) Set(i, j, q int) {
	m[table{i: i, j: j}] = q
}

// A1 = 35 × 15
// A2 = 15 × 5
// A3 = 5 × 10
// A4 = 10 × 20
// A5 = 20 × 25
// p := []int{35, 15, 5, 10, 20, 25}
func matrixChainOrder(p []int) (map[table]int, map[table]int) {
	m := make(mTable)
	s := make(mTable)
	n := len(p)
	for i := 1; i < n; i++ {
		m.Set(i, n, 0)
	}
	for l := 2; l < n; l++ {
		for i := 1; i < n-l+1; i++ {
			j := i + l - 1
			for k := i; k < j; k++ {
				q := m.Get(i, k) + m.Get(k+1, j) + p[i-1]*p[k]*p[j]
				if q < m.Get(i, j) || m.Get(i, j) == 0 {
					m.Set(i, j, q)
					s.Set(i, j, k)
				}
			}
		}
	}
	return m, s
}

// Aa×b⋅Bb×c=Ca×c
func memoizedMatrixChain(p []int) int {
	n := len(p)
	m := make(mTable)
	for i := 1; i < n; i++ {
		for j := i; j < n; j++ {
			m.Set(i, j, math.MaxInt)
		}
	}
	return lookupChain(m, p, 1, n-1)
}

func lookupChain(m mTable, p []int, i, j int) int {
	if m.Get(i, j) < math.MaxInt {
		return m.Get(i, j)
	}
	if i == j {
		m.Set(i, j, 0)
		return m.Get(i, j)
	}
	for k := i; k < j; k++ {
		q := min(
			m.Get(i, j),
			lookupChain(m, p, i, k)+lookupChain(m, p, k+1, j)+p[i-1]*p[k]*p[j],
		)
		m.Set(i, j, q)
	}
	return m.Get(i, j)
}
