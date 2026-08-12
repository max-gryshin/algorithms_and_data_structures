package stdlib

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func main() {
	mux := http.NewServeMux()
	mux.Handle("/hello", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, err := w.Write([]byte("Hello, World!"))
		if err != nil {
			return
		}
	}))
	srv := http.Server{
		Addr:                         ":8080",
		Handler:                      mux,
		DisableGeneralOptionsHandler: false,
		TLSConfig:                    nil,
		ReadHeaderTimeout:            5 * time.Second,
		WriteTimeout:                 30 * time.Second,
		ConnState:                    nil,
		ErrorLog:                     nil,
		BaseContext:                  nil,
		ConnContext:                  nil,
		HTTP2:                        nil,
		Protocols:                    nil,
	}
	errCh := make(chan error, 1)
	go func() {
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errCh <- err
		}
		close(errCh)
	}()

	signCh := make(chan os.Signal, 1)
	signal.Notify(signCh, os.Interrupt, syscall.SIGINT, syscall.SIGTERM)
	select {
	case sig := <-signCh:
		fmt.Println(sig)
	case err := <-errCh:
		fmt.Println(err)
	}

	shutDownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := srv.Shutdown(shutDownCtx); err != nil {
		fmt.Println(err)
		_ = srv.Close()
	}
}
