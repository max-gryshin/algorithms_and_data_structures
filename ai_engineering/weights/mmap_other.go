//go:build !(darwin || linux || freebsd)

package weights

import (
	"errors"
	"os"
)

func mmap(*os.File, int) ([]byte, error) { return nil, errors.New("weights: mmap unsupported") }

func munmap([]byte) error { return nil }
