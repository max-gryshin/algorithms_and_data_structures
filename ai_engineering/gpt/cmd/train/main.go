// Command train trains GoGPT from scratch (lecture L9).
//
//	go run ./ai_engineering/gpt/cmd/train -steps 2000                       # char-level on corpus_ru.txt
//	go run ./ai_engineering/gpt/cmd/train -tokenizer bpe -vocab 400 -steps 2000
//	go run ./ai_engineering/gpt/cmd/train -download shakespeare -d 128 -layers 4 -steps 3000
//
// Pipeline: text → 90/10 split → tokenizer (trained on the train part only) →
// train.bin / val.bin (uint16) → random windows → Trainer. Everything a run writes goes
// to -out: the token files, log.csv and model.ckpt (git-ignored).
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand/v2"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"time"
	"unicode/utf8"

	"algorithms_and_data_structures/ai_engineering/gpt"
	"algorithms_and_data_structures/ai_engineering/gpt/generate"
	"algorithms_and_data_structures/ai_engineering/gpt/train"
	"algorithms_and_data_structures/ai_engineering/mlplm"
	"algorithms_and_data_structures/ai_engineering/tokenizer"
)

// Public-domain corpora fetched at run time (never committed, see data/README.md).
var downloads = map[string]string{
	"shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
}

func main() {
	corpus := flag.String("corpus", "ai_engineering/data/corpus_ru.txt", "UTF-8 text file")
	download := flag.String("download", "", `fetch a corpus first: "shakespeare", "gutenberg:<id>" or an http(s) URL`)
	split := flag.Float64("split", 0.9, "fraction of the text used for training")
	maxChars := flag.Int("max-chars", 0, "use only the first N characters of the training part (scaling experiments)")
	tokKind := flag.String("tokenizer", "char", "char | bpe")
	vocab := flag.Int("vocab", 512, "BPE vocabulary size (bytes + merges)")

	preset := flag.String("preset", "gpt2", "architecture family: gpt2 (learned PE, LayerNorm, GELU) | llama (RoPE, RMSNorm, SwiGLU, GQA)")
	d := flag.Int("d", 64, "model width d")
	layers := flag.Int("layers", 4, "number of blocks L")
	heads := flag.Int("heads", 4, "attention heads h")
	kvHeads := flag.Int("kvheads", 0, "K/V heads (0 = h; llama preset: h/2)")
	seq := flag.Int("ctx", 64, "context length T (training window)")
	dropout := flag.Float64("dropout", 0.1, "dropout probability")

	steps := flag.Int("steps", 2000, "optimizer steps")
	batch := flag.Int("batch", 16, "sequences per micro-batch")
	accum := flag.Int("accum", 1, "gradient accumulation micro-batches per step")
	workers := flag.Int("workers", 1, "data-parallel goroutines")
	threads := flag.Int("threads", 0, "intra-op threads (0 = GOMAXPROCS/workers)")
	lr := flag.Float64("lr", 3e-3, "peak learning rate")
	minLR := flag.Float64("minlr", 0, "final learning rate (0 = lr/10)")
	warmup := flag.Int("warmup", 0, "warmup steps (0 = steps/20)")
	wd := flag.Float64("wd", 0.1, "AdamW weight decay")
	clip := flag.Float64("clip", 1, "gradient norm clip")
	evalEvery := flag.Int("eval-every", 0, "validation every N steps (0 = steps/10)")
	evalTokens := flag.Int("eval-tokens", 0, "validation tokens per evaluation (0 = all)")
	out := flag.String("out", "", "run directory (default ai_engineering/data/runs/<corpus>-<tokenizer>)")
	resume := flag.Bool("resume", false, "continue from <out>/model.ckpt")
	keepBest := flag.Bool("keep-best", true, "checkpoint only when the validation loss improves (early stopping)")
	seed := flag.Uint64("seed", 1, "random seed (init, batches, dropout)")
	sample := flag.Int("sample", 300, "tokens to sample after training")
	temp := flag.Float64("temp", 0.8, "sampling temperature")
	prompt := flag.String("prompt", "", "prompt for the sample (default: first characters of the val part)")
	flag.Parse()

	if *download != "" {
		p, err := fetch(*download, filepath.Dir(*corpus))
		if err != nil {
			log.Fatal(err)
		}
		*corpus = p
	}
	raw, err := os.ReadFile(*corpus)
	if err != nil {
		log.Fatal(err)
	}
	text := string(raw)
	runes := []rune(text)
	cut := int(float64(len(runes)) * *split)
	trainText, valText := string(runes[:cut]), string(runes[cut:])
	if *maxChars > 0 && *maxChars < cut {
		trainText = string(runes[:*maxChars])
	}

	// Tokenizer: trained on the training part only — the validation text must stay unseen.
	var tok train.Tokenizer
	switch *tokKind {
	case "char":
		tok = train.NewCharTokenizer(text) // alphabet of the whole corpus, as in L4 (so val has no unknown chars)
	case "bpe":
		t0 := time.Now()
		tok, err = tokenizer.Train(trainText, *vocab, tokenizer.Options{Pattern: tokenizer.PatternGPT4})
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("BPE: %d tokens trained in %v\n", tok.VocabSize(), time.Since(t0).Round(time.Millisecond))
	default:
		log.Fatalf("unknown tokenizer %q", *tokKind)
	}
	trainIDs, valIDs := tok.Encode(trainText), tok.Encode(valText)
	valChars := utf8.RuneCountInString(valText)
	fmt.Printf("corpus %s: train %d chars → %d tokens, val %d chars → %d tokens (%.2f chars/token), V=%d\n",
		*corpus, utf8.RuneCountInString(trainText), len(trainIDs), valChars, len(valIDs),
		float64(valChars)/float64(len(valIDs)), tok.VocabSize())

	if *out == "" {
		name := strings.TrimSuffix(filepath.Base(*corpus), filepath.Ext(*corpus))
		*out = filepath.Join(filepath.Dir(*corpus), "runs", name+"-"+*tokKind)
	}
	if err := os.MkdirAll(*out, 0o755); err != nil {
		log.Fatal(err)
	}
	// The token files are the interface between data preparation and training: in a real
	// pipeline they are written once (hours of tokenization) and memory-mapped by every run.
	for name, ids := range map[string][]int{"train.bin": trainIDs, "val.bin": valIDs} {
		if err := train.WriteTokens(filepath.Join(*out, name), ids); err != nil {
			log.Fatal(err)
		}
	}
	trainTok, err := train.ReadTokens(filepath.Join(*out, "train.bin"))
	if err != nil {
		log.Fatal(err)
	}
	valTok, err := train.ReadTokens(filepath.Join(*out, "val.bin"))
	if err != nil {
		log.Fatal(err)
	}
	trainSet, valSet := &train.Dataset{Tokens: trainTok}, &train.Dataset{Tokens: valTok}

	bigram := mlplm.BigramLoss(trainIDs, valIDs, tok.VocabSize(), 0.01)
	fmt.Printf("baseline: bigram val %.3f nats/token = %.3f nats/char; uniform ln V = %.3f\n",
		bigram, bigram*float64(len(valIDs)-1)/float64(valChars-1), math.Log(float64(tok.VocabSize())))

	meta, err := train.TokenizerMeta(tok)
	if err != nil {
		log.Fatal(err)
	}
	meta["corpus"] = *corpus
	ckptPath := filepath.Join(*out, "model.ckpt")
	csvFile, err := os.OpenFile(filepath.Join(*out, "log.csv"), os.O_CREATE|os.O_WRONLY|appendOrTrunc(*resume), 0o644)
	if err != nil {
		log.Fatal(err)
	}
	defer csvFile.Close()

	if *evalEvery == 0 {
		*evalEvery = max(1, *steps/10)
	}
	cfg := train.Config{Steps: *steps, Batch: *batch, Seq: *seq, Accum: *accum, Workers: *workers, Threads: *threads,
		MaxLR: *lr, MinLR: *minLR, Warmup: *warmup, WeightDecay: *wd, Clip: *clip,
		EvalEvery: *evalEvery, EvalMaxTokens: *evalTokens, CheckpointPath: ckptPath,
		CheckpointEvery: max(1, *steps/4), KeepBest: *keepBest, Seed: *seed, Meta: meta, CSV: csvFile, Progress: os.Stdout}

	var tr *train.Trainer
	if *resume {
		c, err := gpt.LoadCheckpointFile(ckptPath)
		if err != nil {
			log.Fatal(err)
		}
		if tr, err = train.Resume(c, trainSet, valSet, cfg); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("resumed at step %d\n", tr.Step)
	} else {
		mc := modelConfig(*preset, tok.VocabSize(), *seq, *d, *layers, *heads, *kvHeads, float32(*dropout))
		m := gpt.New(rand.New(rand.NewPCG(*seed, *seed)), mc)
		tr = train.New(m, trainSet, valSet, cfg)
	}
	m := tr.Model
	fmt.Printf("model %s: d=%d L=%d h=%d h_kv=%d ff=%d T=%d dropout=%.2f — %d parameters\n", m.Cfg.Name, m.Cfg.DModel,
		m.Cfg.Layers, m.Cfg.Heads, m.Cfg.KVHeads, m.Cfg.FFHidden, m.Cfg.Context, m.Cfg.Dropout, m.NumParams())
	fmt.Printf("batch %d×%d tokens × accum %d = %d tokens/step, workers %d, threads %d\n",
		*batch, *seq, *accum, *batch**seq**accum, tr.Cfg.Workers, tr.Cfg.Threads)
	if tr.Step == 0 {
		fmt.Printf("val loss at init: %.4f (ln V = %.4f)\n", tr.EvalLoss(), math.Log(float64(tok.VocabSize())))
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	res, err := tr.Run(ctx)
	if err != nil {
		log.Fatal(err)
	}
	perChar := res.ValLoss * float64(len(valIDs)-1) / float64(valChars-1)
	fmt.Printf("done: %d steps in %v (%.0f tok/s); train %.4f, val %.4f nats/token = %.4f nats/char = %.3f bits/char\n",
		res.Steps, res.Elapsed.Round(time.Second), res.TokensPerSec, res.TrainLoss, res.ValLoss, perChar, perChar/math.Ln2)
	if *keepBest && res.BestStep > 0 {
		best := res.BestValLoss * float64(len(valIDs)-1) / float64(valChars-1)
		fmt.Printf("best val %.4f nats/token = %.4f nats/char at step %d (saved); sampling from it\n",
			res.BestValLoss, best, res.BestStep)
		c, err := gpt.LoadCheckpointFile(ckptPath)
		if err != nil {
			log.Fatal(err)
		}
		m = c.Model
	}
	fmt.Printf("checkpoint: %s, log: %s\n", ckptPath, csvFile.Name())

	if *sample > 0 {
		p := *prompt
		if p == "" {
			p = string([]rune(valText)[:min(20, utf8.RuneCountInString(valText))])
		}
		g := generate.Config{MaxTokens: *sample, Sampler: generate.SamplerConfig{Temperature: *temp, Seed: *seed}}
		text, _, err := generate.Text(context.Background(), m, tok, p, g)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("--- sample (temperature %.1f) ---\n%s%s\n", *temp, p, text)
	}
}

func appendOrTrunc(resume bool) int {
	if resume {
		return os.O_APPEND
	}
	return os.O_TRUNC
}

// modelConfig builds a small model of the chosen family.
func modelConfig(preset string, vocab, ctx, d, layers, heads, kvHeads int, dropout float32) gpt.Config {
	var c gpt.Config
	switch preset {
	case "gpt2":
		c = gpt.TinyGPT2(vocab, ctx)
		c.FFHidden = 4 * d
	case "llama":
		c = gpt.TinyLlama(vocab, ctx)
		c.FFHidden = gpt.SwiGLUHidden(d, 8)
		if kvHeads == 0 {
			kvHeads = max(1, heads/2)
		}
	default:
		log.Fatalf("unknown preset %q", preset)
	}
	if kvHeads == 0 {
		kvHeads = heads
	}
	c.Name = fmt.Sprintf("%s-d%d-L%d", preset, d, layers)
	c.DModel, c.Layers, c.Heads, c.KVHeads, c.Dropout = d, layers, heads, kvHeads, dropout
	if err := c.Validate(); err != nil {
		log.Fatal(err)
	}
	return c
}

// fetch downloads a corpus into dir/downloads/ (once) and returns its path.
// Project Gutenberg texts are cut between the "*** START" and "*** END" markers.
func fetch(what, dir string) (string, error) {
	url, name := what, what
	switch {
	case downloads[what] != "":
		url = downloads[what]
	case strings.HasPrefix(what, "gutenberg:"):
		id := strings.TrimPrefix(what, "gutenberg:")
		url = fmt.Sprintf("https://www.gutenberg.org/cache/epub/%s/pg%s.txt", id, id)
		name = "gutenberg-" + id
	case strings.HasPrefix(what, "http://") || strings.HasPrefix(what, "https://"):
		name = strings.TrimSuffix(filepath.Base(what), filepath.Ext(what))
	default:
		return "", fmt.Errorf("unknown download %q", what)
	}
	path := filepath.Join(dir, "downloads", name+".txt")
	if _, err := os.Stat(path); err == nil {
		return path, nil
	}
	fmt.Printf("downloading %s → %s\n", url, path)
	resp, err := http.Get(url)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("GET %s: %s", url, resp.Status)
	}
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	text := strings.ReplaceAll(string(b), "\r\n", "\n")
	if i := strings.Index(text, "*** START"); i >= 0 {
		text = text[i:]
		text = text[strings.Index(text, "\n")+1:]
	}
	if i := strings.Index(text, "*** END"); i >= 0 {
		text = text[:i]
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return "", err
	}
	return path, os.WriteFile(path, []byte(text), 0o644)
}
