package problems

import "strings"

func longestCommonPrefix(strs []string) string {
	prefix := strs[0]
	for i := 1; i < len(strs); i++ {
		for {
			if strings.HasPrefix(strs[i], prefix) {
				break
			} else {
				prefix = prefix[:len(prefix)-1]
				continue
			}
		}
	}
	return prefix
}
