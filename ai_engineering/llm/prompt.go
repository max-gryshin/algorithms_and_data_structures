package llm

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"iter"
	"strings"
	"text/template"
)

// PromptTemplate is a versioned prompt: the system prompt is fixed text (so it can be cached),
// the user message is a text/template filled per request.
// Treat prompts like code: version them, review changes, and run evals on every change (L19).
type PromptTemplate struct {
	Name    string
	Version string
	System  string
	User    *template.Template
}

// NewPromptTemplate parses the user template; it panics on a syntax error like template.Must.
func NewPromptTemplate(name, version, system, user string) *PromptTemplate {
	return &PromptTemplate{Name: name, Version: version, System: system,
		User: template.Must(template.New(name).Option("missingkey=error").Parse(user))}
}

// Render builds a Request with the system prompt marked for caching.
func (p *PromptTemplate) Render(model string, maxTokens int, data any) (Request, error) {
	var sb strings.Builder
	if err := p.User.Execute(&sb, data); err != nil {
		return Request{}, err
	}
	return Request{
		Model: model, MaxTokens: maxTokens, System: p.System, CacheSystem: true,
		Messages: []Message{{Role: RoleUser, Content: sb.String()}},
	}, nil
}

// Fingerprint identifies the exact prompt text, for logs and eval reports:
// two runs with the same fingerprint used byte-identical prompts.
func (p *PromptTemplate) Fingerprint() string {
	h := sha256.New()
	h.Write([]byte(p.System))
	h.Write([]byte{0})
	h.Write([]byte(p.User.Root.String()))
	return p.Name + "@" + p.Version + "#" + hex.EncodeToString(h.Sum(nil))[:12]
}

// WithModelRouter picks the model per request (e.g. a cheaper model for short, simple prompts).
// Note that prompt caches are per model: routing one conversation between models loses cache hits.
func WithModelRouter(c Client, choose func(Request) string) Client {
	return &routed{next: c, choose: choose}
}

type routed struct {
	next   Client
	choose func(Request) string
}

func (r *routed) route(req Request) Request {
	if m := r.choose(req); m != "" {
		req.Model = m
	}
	return req
}

func (r *routed) Complete(ctx context.Context, req Request) (*Response, error) {
	return r.next.Complete(ctx, r.route(req))
}

func (r *routed) Stream(ctx context.Context, req Request) iter.Seq2[Event, error] {
	return r.next.Stream(ctx, r.route(req))
}
