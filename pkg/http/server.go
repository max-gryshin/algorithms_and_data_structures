package http

import (
	rl "algorithms_and_data_structures/pkg/http/middleware"
	"context"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"time"
)

const (
	readTimeout    = 10 * time.Second
	writeTimeout   = 1 * time.Minute
	idleTimeout    = 1 * time.Minute
	maxHeaderBytes = 2 << 32
)

type Server struct {
	srv *http.Server
}

func NewServer(ctx context.Context, port int) *Server {
	srv := &http.Server{
		Addr:           ":" + strconv.Itoa(port),
		Handler:        handlers(ctx),
		ReadTimeout:    readTimeout,
		WriteTimeout:   writeTimeout,
		IdleTimeout:    idleTimeout,
		MaxHeaderBytes: maxHeaderBytes,
	}
	return &Server{
		srv: srv,
	}
}

func handlers(ctx context.Context) *http.ServeMux {
	mux := http.NewServeMux()
	rateLimiter := rl.NewRateLimiter(ctx)
	rateLimitMiddleware := rateLimiter.RateLimiterHandler(helloHandler())

	mux.Handle("/hello", rateLimitMiddleware)

	return mux
}

func helloHandler() http.HandlerFunc {
	return http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
		rw.WriteHeader(http.StatusOK)
		rw.Write([]byte(`{"message": "hello world"}`))
	})
}

func (s *Server) Run(ctx context.Context) error {
	doneCh := make(chan struct{})
	go func() {
		defer close(doneCh)
		if err := s.srv.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
			fmt.Println(fmt.Errorf("error occured in the server, err: %v", err))
		}
	}()

	select {
	case <-ctx.Done():
		err := s.srv.Close()
		<-doneCh

		return err
	case <-doneCh:
		return nil
	}
}

func (s *Server) Close() error {
	return s.srv.Close()
}

func (s *Server) Shutdown(ctx context.Context) error {
	return s.srv.Shutdown(ctx)
}
