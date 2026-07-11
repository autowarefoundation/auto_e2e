// Command api is the DataModelConsole Phase 1 API server: a read-only
// gateway over the platform's S3 datasets/reasoning-label buckets plus
// MLflow and Flyte Admin proxies. See docs/DESIGN.md.
package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/config"
	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/handler"
	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/service"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))
	slog.SetDefault(logger)

	cfg := config.Load()

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	s3svc, err := service.NewS3Service(ctx, cfg.AWSRegion, cfg.DatasetsBucket, cfg.PresignExpiry)
	if err != nil {
		slog.Error("init s3 service", "error", err)
		os.Exit(1)
	}
	mlflowSvc := service.NewMLflowService(cfg.MLflowURL)
	flyteSvc := service.NewFlyteService(cfg.FlyteURL, cfg.FlyteProject, cfg.FlyteDomain)

	healthH := handler.NewHealthHandler(s3svc)
	datasetsH := handler.NewDatasetsHandler(s3svc)
	reasoningH := handler.NewReasoningHandler(s3svc)
	mlflowH := handler.NewMLflowHandler(mlflowSvc)
	flyteH := handler.NewFlyteHandler(flyteSvc)
	statsH := handler.NewStatsHandler(s3svc, mlflowSvc)

	r := chi.NewRouter()
	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(slogRequestLogger)
	r.Use(middleware.Recoverer)
	r.Use(corsMiddleware(cfg.CORSOrigin))

	// Health endpoints are registered OUTSIDE the throttle/timeout stack: when
	// 16 tar scans are in flight, kubelet and the ALB health check must still
	// get an immediate answer, otherwise a throttled probe (429) would restart
	// the pod / drain the target and amplify an overload into an outage.
	r.Get("/healthz", healthH.Healthz)
	r.Get("/readyz", healthH.Readyz)

	r.Route("/api/v1", func(r chi.Router) {
		// Interactive endpoints: bound concurrency (chi Throttle returns 429 on
		// excess) and cap latency at 25s (below CloudFront's 30s origin read
		// timeout so clients get a proper error from us). Health checks above
		// are exempt.
		r.Group(func(r chi.Router) {
			r.Use(middleware.Throttle(16))
			r.Use(middleware.Timeout(25 * time.Second))

			r.Get("/stats", statsH.Get)

			r.Get("/datasets", datasetsH.List)
			r.Get("/datasets/{name}/shards", datasetsH.ListShards)

			r.Get("/reasoning-labels/stats", reasoningH.Stats)
			r.Get("/reasoning-labels/{dataset}/{sample_id}", reasoningH.GetLabel)

			r.Get("/mlflow/experiments", mlflowH.Experiments)
			r.Get("/mlflow/experiments/{id}/runs", mlflowH.Runs)
			r.Get("/mlflow/runs/{id}", mlflowH.Run)
			r.Get("/mlflow/models", mlflowH.Models)

			r.Get("/flyte/executions", flyteH.Executions)
			r.Get("/flyte/executions/{id}", flyteH.Execution)
		})

		// The shard index scans the entire tar once (a multi-hundred-MB / GB
		// read for real shards), so it gets a longer timeout; the result is
		// cached (single-flighted), so only the first request per shard pays it.
		// The player shows a loading state while this warms. Lower throttle
		// since each build is heavy.
		r.Group(func(r chi.Router) {
			r.Use(middleware.Throttle(4))
			r.Use(middleware.Timeout(150 * time.Second))
			r.Get("/datasets/{name}/shards/{shard}/index", datasetsH.GetShardIndex)
			// ListSamples/GetSample each do a full-tar scan identical in cost
			// to the index build, so they belong in the heavy-scan group; in
			// the 25s interactive group they 502 on cold load.
			r.Get("/datasets/{name}/shards/{shard}/samples", datasetsH.ListSamples)
			r.Get("/datasets/{name}/shards/{shard}/samples/{key}", datasetsH.GetSample)
		})

		// Image GETs are cheap bounded range reads and the player fires many in
		// parallel per frame; a looser throttle keeps them from starving against
		// the expensive tar-scan endpoints above (and vice versa).
		r.Group(func(r chi.Router) {
			r.Use(middleware.Throttle(64))
			r.Get("/datasets/{name}/shards/{shard}/samples/{key}/image/{cam}", datasetsH.GetImage)
		})
	})

	srv := &http.Server{
		Addr:              ":" + cfg.Port,
		Handler:           r,
		ReadHeaderTimeout: 10 * time.Second,
		// WriteTimeout must exceed the longest handler timeout (the 150s shard
		// index build) or the server would cut the response mid-build.
		WriteTimeout: 160 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	go func() {
		slog.Info("console api listening",
			"port", cfg.Port,
			"datasets_bucket", cfg.DatasetsBucket,
			"mlflow_url", cfg.MLflowURL,
			"flyte_url", cfg.FlyteURL)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			slog.Error("server failed", "error", err)
			os.Exit(1)
		}
	}()

	<-ctx.Done()
	slog.Info("shutdown signal received, draining connections")

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		slog.Error("graceful shutdown failed", "error", err)
		os.Exit(1)
	}
	slog.Info("server stopped")
}

// slogRequestLogger emits one structured JSON line per request.
func slogRequestLogger(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		ww := middleware.NewWrapResponseWriter(w, r.ProtoMajor)
		next.ServeHTTP(ww, r)
		slog.Info("request",
			"method", r.Method,
			"path", r.URL.Path,
			"status", ww.Status(),
			"bytes", ww.BytesWritten(),
			"duration_ms", time.Since(start).Milliseconds(),
			"request_id", middleware.GetReqID(r.Context()),
			"remote", r.RemoteAddr)
	})
}

// corsMiddleware sets permissive CORS for development ("*") or a fixed
// origin in production (CORS_ORIGIN env). GET-only API, so no preflight
// complexity beyond OPTIONS short-circuit.
func corsMiddleware(origin string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Access-Control-Allow-Origin", origin)
			w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
			if origin != "*" {
				w.Header().Add("Vary", "Origin")
			}
			if r.Method == http.MethodOptions {
				w.WriteHeader(http.StatusNoContent)
				return
			}
			next.ServeHTTP(w, r)
		})
	}
}
