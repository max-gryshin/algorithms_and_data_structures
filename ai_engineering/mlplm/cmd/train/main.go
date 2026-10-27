// Command train fits the Bengio-style character MLP language model of L4 and samples from it.
//
//	go run ./ai_engineering/mlplm/cmd/train -context 3 -hidden 128 -steps 3000
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand/v2"
	"os"

	"algorithms_and_data_structures/ai_engineering/mlplm"
	"algorithms_and_data_structures/ai_engineering/nn"
	"algorithms_and_data_structures/ai_engineering/optim"
)

func main() {
	corpus := flag.String("corpus", "ai_engineering/data/corpus_ru.txt", "UTF-8 text file")
	context := flag.Int("context", 3, "characters of context")
	emb := flag.Int("emb", 10, "embedding size")
	hidden := flag.Int("hidden", 128, "hidden layer width")
	steps := flag.Int("steps", 3000, "training steps")
	batch := flag.Int("batch", 64, "batch size")
	lr := flag.Float64("lr", 3e-3, "peak learning rate")
	wd := flag.Float64("wd", 0.01, "AdamW weight decay")
	temp := flag.Float64("temp", 0.8, "sampling temperature")
	seed := flag.Uint64("seed", 1, "random seed")
	flag.Parse()

	b, err := os.ReadFile(*corpus)
	if err != nil {
		log.Fatal(err)
	}
	v := mlplm.BuildVocab(string(b))
	ids := v.Encode(string(b))
	n := len(ids) * 9 / 10
	train, val := ids[:n], ids[n:]
	fmt.Printf("vocab=%d train=%d val=%d chars\n", v.Size(), len(train), len(val))
	fmt.Printf("bigram baseline val loss: %.3f\n", mlplm.BigramLoss(train, val, v.Size(), 0.01))

	rng := rand.New(rand.NewPCG(*seed, *seed))
	cfg := mlplm.Config{Context: *context, EmbDim: *emb, Hidden: *hidden}
	m := mlplm.NewModel(rng, v.Size(), cfg)
	fmt.Printf("parameters: %d\n", nn.CountParams(m.Parameters()))
	xc, xt := mlplm.Examples(train, cfg.Context)
	vc, vt := mlplm.Examples(val, cfg.Context)

	opt := optim.NewAdamW(m.Parameters(), *lr, *wd)
	sched := optim.CosineSchedule{MaxLR: *lr, MinLR: *lr / 10, WarmupSteps: *steps / 50, TotalSteps: *steps}
	for step := range *steps {
		c, t := mlplm.Batch(rng, xc, xt, *batch)
		opt.ZeroGrad()
		loss := m.Loss(c, t)
		loss.Backward()
		norm := optim.ClipGradNorm(m.Parameters(), 1)
		opt.SetLR(sched.LR(step))
		opt.Step()
		if step%(*steps/10) == 0 || step == *steps-1 {
			fmt.Printf("step %5d  lr %.2e  |g| %.2f  train %.3f  val %.3f\n",
				step, opt.LR(), norm, m.Evaluate(xc, xt), m.Evaluate(vc, vt))
		}
	}
	fmt.Println("---")
	fmt.Println(v.Decode(m.Generate(rng, train[:cfg.Context], 300, float32(*temp))))
}
