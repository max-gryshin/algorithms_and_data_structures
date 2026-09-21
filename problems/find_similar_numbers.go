package problems

// Two numbers are considered similar if:
// - They have the same frequency of each digit.
// - Neither has leading zeros.
// Given two strings representing long integers a and b:
// - If a and b are similar, find the total number of numbers similar to a.
// - If a and b are not similar, find the total number of numbers similar to b.
// Example 1
// a = "1234"
// b = "2341"
func findSimilar(a, b string) int {
	aRunes := []rune(a)
	bRunes := []rune(b)

	aRunesFreq := freq(aRunes)
	bRunesFreq := freq(bRunes)
	isSimilar := true
	for ch, aChFreq := range aRunesFreq {
		bChFreq, ok := bRunesFreq[ch]
		if !ok {
			isSimilar = false
			break
		}
		if aChFreq != bChFreq {
			isSimilar = false
			break
		}
	}
	if isSimilar {
		return total(aRunesFreq, len(aRunes))
	}
	return total(bRunesFreq, len(bRunes))
}

func freq(s []rune) map[rune]int {
	freqMap := make(map[rune]int)
	for _, ch := range s {
		freqMap[ch]++
	}
	return freqMap
}

func factorial(n int) int {
	f := 1
	for i := n; i >= 1; i-- {
		f *= i
	}

	return f
}

// ______n!______ _ _____(n-1)!_____
//
//	c0!c1!..c9!     (c0-1)!c1!..c9!
func total(numbersFreq map[rune]int, numbersLen int) int {
	denominator := 1
	// c0!c1!..c9!
	for _, numberFreq := range numbersFreq {
		denominator *= factorial(numberFreq)
	}

	// n! / c0!c1!..c9!
	total := factorial(numbersLen) / denominator
	zeroCount := numbersFreq['0']
	if zeroCount == 0 {
		return total
	}

	// Replace c0! with (c0-1)!
	denominator = denominator / factorial(zeroCount) * factorial(zeroCount-1)
	// (n-1)! / (c0-1)!c1!..c9!
	withLeadingZero := factorial(numbersLen-1) / denominator

	return total - withLeadingZero
}
