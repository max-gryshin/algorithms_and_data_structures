package problems

import "strings"

func longestCommonPrefix(str []string) string {
	prefix := str[0]
	k := 0
	for i := 1; i < len(str); i++ {
		for {
			if strings.HasPrefix(str[i], prefix) {
				break
			} else {
				prefix = prefix[:len(prefix)-len(string(prefix[k]))]
				continue
			}
		}
	}
	return prefix
}
