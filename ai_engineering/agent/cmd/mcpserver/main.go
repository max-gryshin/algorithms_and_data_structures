// Command mcpserver serves the agent's tools over MCP (stdio transport): calculator,
// list/read/search files and go test inside a sandbox directory, optionally search_docs
// over Markdown notes (BM25 + vectors from L17) and, only with -allow-write, write_file.
//
//	go build -o ~/bin/algo-mcp ./ai_engineering/agent/cmd/mcpserver
//	claude mcp add algo -- ~/bin/algo-mcp -root "$PWD" -docs "$PWD/ai_engineering/lectures"
//
// stdout carries only JSON-RPC messages; all logs go to stderr.
package main

import (
	"context"
	"flag"
	"fmt"
	"io/fs"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"algorithms_and_data_structures/ai_engineering/agent"
	"algorithms_and_data_structures/ai_engineering/agent/mcp"
	"algorithms_and_data_structures/ai_engineering/rag"
)

func main() {
	root := flag.String("root", ".", "sandbox directory for file tools and go test")
	docs := flag.String("docs", "", "directory of .md files to index for search_docs (optional)")
	allowWrite := flag.Bool("allow-write", false, "expose write_file (the host will ask the user before each call)")
	goTest := flag.Bool("go-test", true, "expose go_test (runs test code from the sandbox: trust the directory)")
	flag.Parse()
	log.SetOutput(os.Stderr) // stdout is the protocol channel
	log.SetPrefix("mcpserver: ")

	sb, err := agent.NewSandbox(*root)
	if err != nil {
		log.Fatal(err)
	}
	tools := []agent.Tool{agent.Calculator(), agent.ListFilesTool(sb), agent.ReadFileTool(sb), agent.SearchFilesTool(sb)}
	if *goTest {
		tools = append(tools, agent.GoTestTool(sb, 2*time.Minute))
	}
	if *allowWrite {
		tools = append(tools, agent.WriteFileTool(sb))
	}
	if *docs != "" {
		store, n, err := indexDocs(*docs)
		if err != nil {
			log.Fatal(err)
		}
		log.Printf("indexed %d chunks from %s", n, *docs)
		tools = append(tools, agent.RAGSearchTool(store))
	}

	srv := &mcp.Server{
		Info:           mcp.Implementation{Name: "algo-course", Version: "0.1.0"},
		Instructions:   fmt.Sprintf("Tools over the Go project at %s: search and read code, run go test, search lecture notes.", sb.Root),
		Tools:          agent.NewRegistry(tools...),
		AllowDangerous: *allowWrite,
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	log.Printf("serving %d tools, sandbox %s", len(tools), sb.Root)
	if err := srv.Serve(ctx, os.Stdin, os.Stdout); err != nil {
		log.Fatal(err)
	}
}

// indexDocs chunks every Markdown file under dir into a hybrid store. HashEmbedder keeps
// it offline; swap in a real embedding model for semantic search.
func indexDocs(dir string) (*rag.Store, int, error) {
	store := rag.NewStore(rag.HashEmbedder{Dim: 512}, rag.NewHNSW(16, 100, 64, 1))
	var chunks []rag.Chunk
	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(path, ".md") {
			return err
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(dir, path)
		chunks = append(chunks, rag.ChunkMarkdown(rel, string(data), 1500, 200)...)
		return nil
	})
	if err != nil {
		return nil, 0, err
	}
	return store, len(chunks), store.Add(context.Background(), chunks...)
}
