package serve

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"algorithms_and_data_structures/ai_engineering/gpt/generate"
)

// Server is an OpenAI-compatible HTTP API over an Engine:
//
//	POST /v1/chat/completions   (stream: SSE "data: {chunk}" … "data: [DONE]")
//	GET  /v1/models
//	GET  /metrics
//
// llm.OpenAICompatible from L16 talks to it unchanged.
type Server struct {
	Engine *Engine
	Tok    generate.Tokenizer
	Model  string
	EOS    []int
	// Template renders the conversation into the prompt; nil = "role: content\n" lines + "assistant: ".
	Template func(msgs []ChatMessage) string
	ids      atomic.Int64
}

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Model         string        `json:"model"`
	Messages      []ChatMessage `json:"messages"`
	MaxTokens     int           `json:"max_tokens"`
	Temperature   *float64      `json:"temperature"`
	TopP          float64       `json:"top_p"`
	Seed          uint64        `json:"seed"`
	Stop          []string      `json:"stop"`
	Stream        bool          `json:"stream"`
	StreamOptions struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options"`
}

type usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", s.chat)
	mux.HandleFunc("GET /v1/models", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, 200, map[string]any{"object": "list", "data": []any{
			map[string]any{"id": s.Model, "object": "model", "owned_by": "gogpt"}}})
	})
	mux.HandleFunc("GET /metrics", func(w http.ResponseWriter, r *http.Request) { s.Engine.Metrics.WriteText(w) })
	return mux
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("content-type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v)
}

func apiError(w http.ResponseWriter, code int, msg string) {
	if code == http.StatusTooManyRequests {
		w.Header().Set("retry-after", "1")
	}
	writeJSON(w, code, map[string]any{"error": map[string]any{"message": msg, "type": http.StatusText(code)}})
}

func (s *Server) prompt(msgs []ChatMessage) string {
	if s.Template != nil {
		return s.Template(msgs)
	}
	var sb strings.Builder
	for _, m := range msgs {
		fmt.Fprintf(&sb, "%s: %s\n", m.Role, m.Content)
	}
	sb.WriteString("assistant: ")
	return sb.String()
}

func (s *Server) chat(w http.ResponseWriter, r *http.Request) {
	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		apiError(w, 400, "bad json: "+err.Error())
		return
	}
	sc := generate.SamplerConfig{Temperature: 1, TopP: req.TopP, Seed: req.Seed}
	if req.Temperature != nil {
		sc.Temperature = *req.Temperature
	}
	ctx, cancel := context.WithCancel(r.Context()) // client gone or stop sequence → free the KV
	defer cancel()
	events, err := s.Engine.Submit(ctx, Request{Prompt: s.Tok.Encode(s.prompt(req.Messages)),
		MaxTokens: req.MaxTokens, Sampler: sc, EOS: s.EOS})
	switch {
	case errors.Is(err, ErrOverloaded):
		apiError(w, 429, err.Error())
		return
	case err != nil:
		apiError(w, 400, err.Error())
		return
	}
	id := fmt.Sprintf("chatcmpl-%d", s.ids.Add(1))
	created := time.Now().Unix()
	chunk := func(delta map[string]any, finish any) map[string]any {
		return map[string]any{"id": id, "object": "chat.completion.chunk", "created": created, "model": s.Model,
			"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}}}
	}
	var flusher http.Flusher
	send := func(v any) {
		b, _ := json.Marshal(v)
		fmt.Fprintf(w, "data: %s\n\n", b)
		if flusher != nil {
			flusher.Flush()
		}
	}
	if req.Stream {
		flusher, _ = w.(http.Flusher)
		w.Header().Set("content-type", "text/event-stream")
		w.Header().Set("cache-control", "no-cache")
		send(chunk(map[string]any{"role": "assistant", "content": ""}, nil))
	}
	var detok generate.Detokenizer
	var full, pending strings.Builder // pending: text held back as a possible stop-sequence prefix
	emit := func(text string) {
		if text == "" {
			return
		}
		full.WriteString(text)
		if req.Stream {
			send(chunk(map[string]any{"content": text}, nil))
		}
	}
	finish, u := "length", usage{}
	for ev := range events {
		if ev.Finish != "" {
			u = usage{ev.PromptTokens, ev.CompletionTokens, ev.PromptTokens + ev.CompletionTokens}
			if ev.Finish == "stop" || ev.Finish == "length" {
				finish = ev.Finish
			}
			break
		}
		pending.WriteString(detok.Push(s.Tok.TokenBytes(ev.Token)))
		text, stopped, hold := splitStop(pending.String(), req.Stop)
		emit(text)
		pending.Reset()
		pending.WriteString(hold)
		if stopped {
			finish = "stop"
			cancel()
			for ev := range events { // drain until the engine reports usage
				if ev.Finish != "" {
					u = usage{ev.PromptTokens, ev.CompletionTokens, ev.PromptTokens + ev.CompletionTokens}
				}
			}
			break
		}
	}
	if finish != "stop" {
		emit(pending.String() + detok.Flush())
	}
	if !req.Stream {
		writeJSON(w, 200, map[string]any{"id": id, "object": "chat.completion", "created": created, "model": s.Model,
			"choices": []any{map[string]any{"index": 0, "finish_reason": finish,
				"message": map[string]any{"role": "assistant", "content": full.String()}}},
			"usage": u})
		return
	}
	send(chunk(map[string]any{}, finish))
	if req.StreamOptions.IncludeUsage {
		send(map[string]any{"id": id, "object": "chat.completion.chunk", "created": created, "model": s.Model,
			"choices": []any{}, "usage": u})
	}
	fmt.Fprint(w, "data: [DONE]\n\n")
	if flusher != nil {
		flusher.Flush()
	}
}

// splitStop returns the text safe to emit, whether a stop sequence occurred, and the
// tail to hold back because it may be the beginning of a stop sequence.
func splitStop(text string, stops []string) (emit string, stopped bool, hold string) {
	for _, st := range stops {
		if st != "" {
			if i := strings.Index(text, st); i >= 0 {
				return text[:i], true, ""
			}
		}
	}
	keep := 0
	for _, st := range stops {
		for k := min(len(st)-1, len(text)); k > keep; k-- {
			if strings.HasSuffix(text, st[:k]) {
				keep = k
				break
			}
		}
	}
	return text[:len(text)-keep], false, text[len(text)-keep:]
}
