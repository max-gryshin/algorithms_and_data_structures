package hashmap

import "sort"

func topKFreqElems(elems []int, k int) []int {
	res := make([]int, 0, k)
	m := make(map[int]int)
	for _, e := range elems {
		m[e]++
	}
	type pair struct {
		key, val int
	}
	pairs := make([]pair, 0, len(elems))
	for k, v := range m {
		pairs = append(pairs, pair{k, v})
	}
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].val > pairs[j].val
	})

	for i := 0; i < k; i++ {
		res = append(res, pairs[i].key)
	}

	return res
}
