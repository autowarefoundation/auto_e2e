package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/config"
	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/service"
	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/store"
)

func runReasoningMaterializer(
	ctx context.Context,
	cfg *config.Config,
	args []string,
	output io.Writer,
) error {
	flags := flag.NewFlagSet("materialize-reasoning", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	dataset := flags.String("dataset", "", "published dataset name")
	version := flags.String("version", "", "immutable dataset version")
	manifestSHA256 := flags.String(
		"manifest-sha256",
		"",
		"expected immutable publication manifest SHA-256",
	)
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *dataset == "" ||
		*version == "" ||
		*manifestSHA256 == "" ||
		flags.NArg() != 0 {
		return fmt.Errorf(
			"--dataset, --version, and --manifest-sha256 are required",
		)
	}
	if !service.ValidVersion(*version) {
		return fmt.Errorf("invalid dataset version %q", *version)
	}
	if !store.ValidReasoningGeneration(*manifestSHA256) {
		return fmt.Errorf("invalid publication manifest SHA-256")
	}

	dynamoStore, err := store.New(ctx, cfg.AWSRegion, cfg.DynamoTable)
	if err != nil {
		return fmt.Errorf("initialize dynamo store: %w", err)
	}
	s3Service, err := service.NewS3Service(
		ctx,
		cfg.AWSRegion,
		cfg.DatasetsBucket,
		cfg.PresignExpiry,
		dynamoStore,
		cfg.ArtifactsBucket,
	)
	if err != nil {
		return fmt.Errorf("initialize S3 service: %w", err)
	}
	if !s3Service.ValidDataset(*dataset) {
		return fmt.Errorf("unknown dataset %q", *dataset)
	}
	response, err := s3Service.MaterializeReasoning(
		ctx, *dataset, *version, *manifestSHA256,
	)
	if err != nil {
		return err
	}
	if err := json.NewEncoder(output).Encode(response); err != nil {
		return fmt.Errorf("encode materialization result: %w", err)
	}
	return nil
}
