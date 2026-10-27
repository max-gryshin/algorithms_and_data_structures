// Command server serves a GoGPT checkpoint over an OpenAI-compatible API (lecture L14).
//
//	go run ./ai_engineering/serve/cmd/server -ckpt ai_engineering/data/runs/corpus_ru-char/model.ckpt
//	go run ./ai_engineering/llm/cmd/chat -provider openai -base-url http://localhost:8080 -model gogpt
package main

import (
	"context"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"

	"algorithms_and_data_structures/ai_engineering/gpt"
	"algorithms_and_data_structures/ai_engineering/gpt/train"
	"algorithms_and_data_structures/ai_engineering/serve"
)

func main() {
	ckpt := flag.String("ckpt", "ai_engineering/data/runs/corpus_ru-char/model.ckpt", "checkpoint written by gpt/cmd/train")
	addr := flag.String("addr", ":8080", "listen address")
	static := flag.Bool("static", false, "static batching instead of continuous")
	seqs := flag.Int("seqs", 16, "max running sequences")
	blocks := flag.Int("blocks", 2048, "KV blocks of 16 tokens")
	chunk := flag.Int("chunk", 64, "prefill chunk (0 = whole prompt)")
	flag.Parse()

	c, err := gpt.LoadCheckpointFile(*ckpt)
	if err != nil {
		log.Fatal(err)
	}
	tok, err := train.TokenizerFromMeta(c.Meta)
	if err != nil {
		log.Fatal(err)
	}
	cfg := serve.EngineConfig{BlockSize: 16, NumBlocks: *blocks, MaxSeqs: *seqs, PrefillChunk: *chunk, MaxWaiting: 256}
	if *static {
		cfg.Policy = serve.Static
	}
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()
	e := serve.NewEngine(c.Model, cfg)
	go e.Run(ctx)
	srv := &http.Server{Addr: *addr, Handler: (&serve.Server{Engine: e, Tok: tok, Model: "gogpt"}).Handler()}
	go func() { <-ctx.Done(); srv.Close() }()
	log.Printf("serving %s on %s", c.Model.Cfg.Name, *addr)
	if err := srv.ListenAndServe(); err != http.ErrServerClosed {
		log.Fatal(err)
	}
}
