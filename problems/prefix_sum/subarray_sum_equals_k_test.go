package prefix_sum

import "testing"

func TestSubarraySumK(t *testing.T) {
	table := []struct {
		array []int
		k     int
		want  int
	}{
		{array: []int{1, 1, 1}, k: 2, want: 2},
		{array: []int{1, -1, 0}, k: 0, want: 3},
		{array: []int{4, 0, 4}, k: 4, want: 4},
	}

	for _, tc := range table {
		got := subarraySumK(tc.array, tc.k)
		if got != tc.want {
			t.Errorf("subarray_sum_k(%v, %d) = %d, want %d", tc.array, tc.k, got, tc.want)
		}
	}
}
