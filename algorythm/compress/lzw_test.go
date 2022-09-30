package compress

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func check(e error) {
	if e != nil {
		panic(e)
	}
}

// There is a problem with some symbols: … » ’ ’ —
func TestCompressLZWest(t *testing.T) {
	path, errPath := filepath.Abs("lotr.txt")
	check(errPath)
	s, err := os.ReadFile(path)
	check(err)
	fmt.Printf("Size of file before compression: %d kB \n", len(s)/1000)
	stringToCompress := string(s)
	resCompress := CompressLZWChunks(stringToCompress)
	sizeOfCompressedData := 0
	for _, t := range resCompress {
		sizeOfCompressedData += len(t)
	}
	fmt.Printf("Size of compressed data: %d kB \n", sizeOfCompressedData/1000)
	start := time.Now()
	resultOfDecompress := DecompressConcurrentLZW(resCompress)
	elapsed := time.Since(start)
	fmt.Printf("Decompresson took  %f seconds \n", elapsed.Seconds())
	if stringToCompress != resultOfDecompress {
		t.Error("Strings are not equal")
	}
}
