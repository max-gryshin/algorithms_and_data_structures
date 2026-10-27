package rag

import "math"

// RecallAtK is the share of relevant documents found in the first k results.
func RecallAtK(retrieved []int, relevant map[int]bool, k int) float64 {
	if len(relevant) == 0 {
		return 0
	}
	found := 0
	for _, d := range retrieved[:min(k, len(retrieved))] {
		if relevant[d] {
			found++
		}
	}
	return float64(found) / float64(len(relevant))
}

// ReciprocalRank is 1/rank of the first relevant result (0 if none).
// MRR is its mean over many queries.
func ReciprocalRank(retrieved []int, relevant map[int]bool) float64 {
	for i, d := range retrieved {
		if relevant[d] {
			return 1 / float64(i+1)
		}
	}
	return 0
}

// NDCGAtK is the normalized discounted cumulative gain:
//
//	DCG@k = Σ_{i=1..k} rel_i / log2(i + 1),  nDCG@k = DCG@k / IDCG@k,
//
// where IDCG is the DCG of the ideal ordering. gain maps a document to its graded
// relevance (0 = irrelevant).
func NDCGAtK(retrieved []int, gain map[int]float64, k int) float64 {
	dcg := 0.0
	for i, d := range retrieved[:min(k, len(retrieved))] {
		dcg += gain[d] / math.Log2(float64(i+2))
	}
	ideal := make([]float64, 0, len(gain))
	for _, g := range gain {
		ideal = append(ideal, g)
	}
	// sort descending (insertion sort: gain lists are tiny)
	for i := 1; i < len(ideal); i++ {
		for j := i; j > 0 && ideal[j] > ideal[j-1]; j-- {
			ideal[j], ideal[j-1] = ideal[j-1], ideal[j]
		}
	}
	idcg := 0.0
	for i, g := range ideal[:min(k, len(ideal))] {
		idcg += g / math.Log2(float64(i+2))
	}
	if idcg == 0 {
		return 0
	}
	return dcg / idcg
}
