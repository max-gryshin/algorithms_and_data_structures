// Command gptcalc prints parameters, FLOPs and memory for GPT-2 124M, Llama-2-7B
// and Llama-3-8B (lecture L8):
//
//	go run ./ai_engineering/gpt/cmd/gptcalc -tokens 15e12
package main

import (
	"flag"
	"fmt"

	"algorithms_and_data_structures/ai_engineering/gpt"
)

func main() {
	tokens := flag.Float64("tokens", 1e12, "training tokens D for the 6·N·D estimate")
	flag.Parse()
	for _, c := range []gpt.Config{gpt.GPT2Small, gpt.Llama2_7B, gpt.Llama3_8B} {
		fmt.Println(c.Report(*tokens))
	}
}
