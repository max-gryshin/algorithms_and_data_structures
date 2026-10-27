// Command chat is a terminal chat over the llm package: streaming output, conversation history,
// retries, rate limiting, a circuit breaker and a cost meter.
//
//	export ANTHROPIC_API_KEY=...
//	go run ./ai_engineering/llm/cmd/chat -model claude-opus-5-5
//	go run ./ai_engineering/llm/cmd/chat -provider openai -base-url http://localhost:8080 -model gogpt
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"time"

	"algorithms_and_data_structures/ai_engineering/llm"
)

func main() {
	provider := flag.String("provider", "anthropic", "anthropic | openai (any OpenAI-compatible server)")
	baseURL := flag.String("base-url", "", "API base URL (default: provider's)")
	model := flag.String("model", "claude-opus-5-5", "model id")
	system := flag.String("system", "You are a concise assistant. Answer in the user's language.", "system prompt")
	maxTokens := flag.Int("max-tokens", 4096, "max output tokens per answer")
	budget := flag.Float64("budget", 1.0, "stop after spending this many USD (0 = unlimited)")
	flag.Parse()

	var base llm.Client
	switch *provider {
	case "anthropic":
		a, err := llm.NewAnthropicFromEnv()
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		if *baseURL != "" {
			a.BaseURL = *baseURL
		}
		base = a
	case "openai":
		base = &llm.OpenAICompatible{APIKey: os.Getenv("OPENAI_API_KEY"), BaseURL: *baseURL}
	default:
		fmt.Fprintln(os.Stderr, "unknown provider", *provider)
		os.Exit(2)
	}

	// Middleware order, outside → in: meter → breaker → rate limit → retry → provider.
	meter := &llm.Meter{Budget: *budget}
	client := llm.WithMeter(
		llm.WithBreaker(
			llm.WithRateLimit(llm.WithRetry(base, llm.DefaultRetryPolicy), llm.NewTokenBucket(5, 50), nil),
			llm.NewBreaker(3, 30*time.Second)),
		meter)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	var history []llm.Message
	in := bufio.NewScanner(os.Stdin)
	fmt.Println("Ctrl+D to exit.")
	for {
		fmt.Print("\n> ")
		if !in.Scan() {
			break
		}
		q := strings.TrimSpace(in.Text())
		if q == "" {
			continue
		}
		history = append(history, llm.Message{Role: llm.RoleUser, Content: q})
		req := llm.Request{Model: *model, System: *system, CacheSystem: true, Messages: history, MaxTokens: *maxTokens}

		var answer strings.Builder
		var stopReason string
		var failed error
		for ev, err := range client.Stream(ctx, req) {
			if err != nil {
				failed = err
				break
			}
			switch ev.Type {
			case llm.EventTextDelta:
				fmt.Print(ev.Text)
				answer.WriteString(ev.Text)
			case llm.EventStop:
				stopReason = ev.StopReason
			}
		}
		fmt.Println()
		if failed != nil {
			fmt.Fprintln(os.Stderr, "error:", failed)
			history = history[:len(history)-1] // drop the unanswered question
			if ctx.Err() != nil {
				break
			}
			continue
		}
		if stopReason == "max_tokens" || stopReason == "length" {
			fmt.Fprintln(os.Stderr, "[answer truncated by max_tokens]")
		}
		history = append(history, llm.Message{Role: llm.RoleAssistant, Content: answer.String()})
		fmt.Fprintf(os.Stderr, "[%s · %s]\n", stopReason, meter)
	}
	fmt.Fprintf(os.Stderr, "\ntotal: %s\n", meter)
}
