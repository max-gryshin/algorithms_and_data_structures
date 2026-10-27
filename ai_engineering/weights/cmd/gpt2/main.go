// Command gpt2 runs OpenAI GPT-2 124M from the Hugging Face files in pure Go (L12).
//
//	mkdir -p ~/models/gpt2 && cd ~/models/gpt2
//	for f in config.json vocab.json merges.txt model.safetensors; do
//	  curl -LO https://huggingface.co/openai-community/gpt2/resolve/main/$f
//	done
//	go run ./ai_engineering/weights/cmd/gpt2 -dir ~/models/gpt2 -prompt "The meaning of life is"
//
// -download fetches the four files into -dir itself; -quant int8|int4 fake-quantizes
// the weights; -ppl file prints the perplexity of a text file.
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"algorithms_and_data_structures/ai_engineering/gpt/generate"
	"algorithms_and_data_structures/ai_engineering/tokenizer"
	"algorithms_and_data_structures/ai_engineering/weights"
)

func main() {
	dir := flag.String("dir", "gpt2", "directory with config.json, vocab.json, merges.txt, model.safetensors")
	download := flag.Bool("download", false, "download the files from Hugging Face into -dir")
	prompt := flag.String("prompt", "The meaning of life is", "prompt")
	n := flag.Int("n", 50, "tokens to generate")
	temp := flag.Float64("temp", 0, "temperature (0 = greedy)")
	topK := flag.Int("topk", 40, "top-k")
	seed := flag.Uint64("seed", 1, "sampler seed")
	quant := flag.String("quant", "f32", "f32 | int8 | int4 (fake quantization)")
	ppl := flag.String("ppl", "", "text file to measure perplexity on (first 1024 tokens)")
	flag.Parse()

	if *download {
		if err := fetch(*dir); err != nil {
			log.Fatal(err)
		}
	}
	t0 := time.Now()
	m, err := weights.LoadGPT2Dir(*dir)
	if err != nil {
		log.Fatal(err)
	}
	tk, err := tokenizer.LoadGPT2Dir(*dir)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Fprintf(os.Stderr, "loaded %s: %.1fM params in %v\n", m.Cfg.Name, float64(m.NumParams())/1e6, time.Since(t0).Round(time.Millisecond))
	if *quant != "f32" {
		q, f := weights.FakeQuantize(m, weights.Format(*quant))
		fmt.Fprintf(os.Stderr, "%s: matrices %.0f MB → %.0f MB\n", *quant, float64(f)/1e6, float64(q)/1e6)
	}
	if *ppl != "" {
		b, err := os.ReadFile(*ppl)
		if err != nil {
			log.Fatal(err)
		}
		ids := tk.Encode(string(b))
		ids = ids[:min(len(ids), m.Cfg.Context+1)]
		fmt.Printf("perplexity (%d tokens): %.3f\n", len(ids), weights.Perplexity(m, ids))
		return
	}
	eot, _ := tk.SpecialID(tokenizer.EndOfText)
	cfg := generate.Config{MaxTokens: *n, EOS: []int{eot},
		Sampler: generate.SamplerConfig{Temperature: *temp, TopK: *topK, Seed: *seed}}
	fmt.Print(*prompt)
	start := time.Now()
	toks := 0
	for t, err := range generate.Generate(context.Background(), m, tk, *prompt, cfg) {
		if err != nil {
			log.Fatal(err)
		}
		fmt.Print(t.Text)
		toks++
	}
	el := time.Since(start)
	fmt.Fprintf(os.Stderr, "\n%d tokens in %v: %.1f tokens/s\n", toks, el.Round(time.Millisecond), float64(toks)/el.Seconds())
}

func fetch(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	for _, f := range []string{"config.json", "vocab.json", "merges.txt", "model.safetensors"} {
		path := filepath.Join(dir, f)
		if _, err := os.Stat(path); err == nil {
			continue
		}
		fmt.Fprintf(os.Stderr, "downloading %s…\n", f)
		resp, err := http.Get("https://huggingface.co/openai-community/gpt2/resolve/main/" + f)
		if err != nil {
			return err
		}
		if resp.StatusCode != http.StatusOK {
			resp.Body.Close()
			return fmt.Errorf("%s: %s", f, resp.Status)
		}
		out, err := os.Create(path + ".part")
		if err == nil {
			_, err = io.Copy(out, resp.Body)
			out.Close()
		}
		resp.Body.Close()
		if err != nil {
			return err
		}
		if err := os.Rename(path+".part", path); err != nil {
			return err
		}
	}
	return nil
}
