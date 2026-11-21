// Command figs draws lecture figures as SVG from the course code itself: every number
// on a picture is computed by autograd, not typed in by hand.
//
//	go run ./ai_engineering/lectures/figs                 # from the repository root
//	go run ./ai_engineering/lectures/figs -out some/dir   # anywhere else
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"slices"
)

var figures = map[string]func() *canvas{
	"L03_lm_head_graph.svg":   l03LMHeadGraph,
	"L03_matmul_grad.svg":     l03MatMulGrad,
	"L03_softmax_ce_step.svg": l03SoftmaxCEStep,
	"L03_gradcheck_eps.svg":   l03GradCheckEps,
}

func main() {
	out := flag.String("out", "ai_engineering/lectures/img", "directory to write the SVG files to")
	flag.Parse()

	names := make([]string, 0, len(figures))
	for name := range figures {
		names = append(names, name)
	}
	slices.Sort(names)
	for _, name := range names {
		path := filepath.Join(*out, name)
		if err := os.WriteFile(path, []byte(figures[name]().String()), 0o644); err != nil {
			log.Fatal(err)
		}
		fmt.Println(path)
	}
}
