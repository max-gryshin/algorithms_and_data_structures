package problems

func romanToInt(s string) int {
	vocabulary := map[string]int{
		"I": 1,
		"V": 5,
		"X": 10,
		"L": 50,
		"C": 100,
		"D": 500,
		"M": 1000,
	}
	res := 0
	num := 0
	ok := false
	var prev string
	for _, ch := range []rune(s) {
		if num, ok = vocabulary[string(ch)]; ok == true {
			if (prev == "I" && (string(ch) == "V" || string(ch) == "X")) ||
				(prev == "X" && (string(ch) == "L" || string(ch) == "C")) ||
				prev == "C" && (string(ch) == "D" || string(ch) == "M") {
				res = res + num - vocabulary[prev]*2
				prev = ""
				continue
			}
			res += num
			prev = string(ch)
		}
	}
	return res
}
