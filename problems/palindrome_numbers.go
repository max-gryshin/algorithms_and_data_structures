package problems

func isPalindromeOwn(x int) bool {
	if x < 0 {
		return false
	}
	if x == 0 {
		return true
	}
	var p []int
	var c int
	step := 10
	modulus := step
	m := 1
	for i := x; i > 0; {
		if i%modulus == 0 {
			p = append(p, 0)
			modulus *= step
			c = 0
			m *= step
			continue
		}
		c += m
		i -= m
		if i%modulus == 0 {
			p = append(p, c/m)
			modulus *= step
			c = 0
			m *= step
		}
	}
	k := len(p)
	for j := 0; j < k; j++ {
		if j == k-1 || j > (k-1) {
			break
		}
		if p[j] != p[k-1] {
			return false
		}
		k--
	}

	return true
}

func isPalindromeLeetCode(x int) bool {
	if x < 0 || (x%10 == 0 && x != 0) {
		return false
	}
	revertedNumber := 0
	for x > revertedNumber {
		revertedNumber = revertedNumber*10 + x%10
		x /= 10
	}
	return x == revertedNumber || x == revertedNumber/10
}
