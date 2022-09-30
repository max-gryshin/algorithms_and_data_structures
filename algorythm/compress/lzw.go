// Package compress

package compress

import (
	"fmt"
	"runtime"
	"sync"
)

func CompressLZW(s string) []int {
	code := 256
	d := make(map[string]int)
	for i := 0; i < 256; i++ {
		d[string(rune(i))] = i
	}
	var currChar string
	var result []int
	for _, c := range []byte(s) {
		if _, isTrue := d[currChar+string(c)]; isTrue {
			currChar += string(c)
		} else {
			result = append(result, d[currChar])
			d[currChar+string(c)] = code
			code++
			currChar = string(c)
		}
	}
	if currChar != "" {
		result = append(result, d[currChar])
	}
	return result
}

func DecompressLZW(compressed []int) string {
	var code = 256
	var entry string
	dictionary := make(map[int]string)
	for i := 0; i < 256; i++ {
		dictionary[i] = string(rune(i))
	}
	char := string(rune(compressed[0]))
	res := char
	for _, v := range compressed[1:] {
		if x, ok := dictionary[v]; ok {
			entry = x
		} else if v == code {
			entry = char + char[:1]
		} else {
			panic(fmt.Sprintf("Bad compressed element: %dictionary", v))
		}
		res += entry
		dictionary[code] = char + entry[:1]
		code++
		char = entry
	}
	return res
}

func CompressLZWChunks(s string) map[int][]int {
	resultM := make(map[int][]int)
	sBytes := []byte(s)
	rangeBytes := countRange(len(sBytes))
	for i := 0; i < len(rangeBytes)-1; i++ {
		resultM[i] = CompressLZW(string(sBytes[rangeBytes[i]:rangeBytes[i+1]]))
	}
	return resultM
}

func DecompressConcurrentLZW(compressed map[int][]int) string {
	resultMap := make(map[int]string)
	var result string
	wt := sync.WaitGroup{}
	compressedLen := len(compressed)
	for k, c := range compressed {
		wt.Add(1)
		go func(waitGr *sync.WaitGroup, compressedString []int, res map[int]string, i int) {
			res[i] += DecompressLZW(compressedString)
			defer func() {
				wt.Done()
			}()
		}(&wt, c, resultMap, k)
	}
	wt.Wait()
	for i := 0; i < compressedLen; i++ {
		result += resultMap[i]
	}

	return result
}

func countRange(cLen int) []int {
	var res []int
	numCPU := runtime.NumCPU()
	res = append(res, 0)
	count := 0
	for {
		if count >= cLen {
			res = append(res, cLen)
			break
		}
		count += cLen / numCPU
		if count >= cLen {
			res = append(res, cLen)
			break
		}
		res = append(res, count)
	}
	return res
}
