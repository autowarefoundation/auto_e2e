package service

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/s3"
)

type cachedCompatibleVersions struct {
	versions []string
	at       time.Time
}

// compatibleArtifactVersions returns newest-first publication coordinates
// whose scene and sample inventory matches the requested dataset pack. This
// carry-forward policy is deliberately limited to canonical KITScenes.
func (s *S3Service) compatibleArtifactVersions(
	ctx context.Context,
	dataset string,
	requestedVersion string,
) (string, []string, error) {
	targetVersion, err := s.publishedVersion(
		ctx, dataset, requestedVersion,
	)
	if err != nil {
		return "", nil, err
	}
	if dataset != "kitscenes" ||
		!requiresPublicationManifest(targetVersion) {
		return targetVersion, []string{targetVersion}, nil
	}

	cacheKey := dataset + "/" + targetVersion
	s.compatibleVersionMu.Lock()
	if cached, ok := s.compatibleVersionCache[cacheKey]; ok &&
		time.Since(cached.at) < versionTTL {
		versions := append([]string(nil), cached.versions...)
		s.compatibleVersionMu.Unlock()
		return targetVersion, versions, nil
	}
	s.compatibleVersionMu.Unlock()

	target, err := s.loadPublicationManifest(
		ctx, dataset, targetVersion,
	)
	if err != nil {
		return "", nil, err
	}
	versions, err := s.publishedVersionNames(ctx, dataset)
	if err != nil {
		return "", nil, err
	}
	compatible := make([]string, 0, len(versions))
	for _, version := range versions {
		if !requiresPublicationManifest(version) {
			continue
		}
		manifest, err := s.loadPublicationManifest(
			ctx, dataset, version,
		)
		if err != nil {
			continue
		}
		if sameSceneInventory(target, manifest) {
			compatible = append(compatible, version)
		}
	}
	if len(compatible) == 0 {
		return "", nil, fmt.Errorf(
			"no compatible publication for %s/%s",
			dataset,
			targetVersion,
		)
	}

	s.compatibleVersionMu.Lock()
	s.compatibleVersionCache[cacheKey] = cachedCompatibleVersions{
		versions: append([]string(nil), compatible...),
		at:       nowFunc(),
	}
	s.compatibleVersionMu.Unlock()
	return targetVersion, compatible, nil
}

func (s *S3Service) publishedVersionNames(
	ctx context.Context,
	dataset string,
) ([]string, error) {
	prefix := dataset + "/"
	out, err := s.client.ListObjectsV2(ctx, &s3.ListObjectsV2Input{
		Bucket:    aws.String(s.bucket),
		Prefix:    aws.String(prefix),
		Delimiter: aws.String("/"),
	})
	if err != nil {
		return nil, fmt.Errorf(
			"list artifact source versions for %s: %w",
			dataset,
			err,
		)
	}
	versions := make([]string, 0, len(out.CommonPrefixes))
	for _, commonPrefix := range out.CommonPrefixes {
		version := strings.TrimSuffix(
			strings.TrimPrefix(
				aws.ToString(commonPrefix.Prefix),
				prefix,
			),
			"/",
		)
		if isVersionDir(version) {
			versions = append(versions, version)
		}
	}
	sortVersionsNewestFirst(versions)
	return versions, nil
}

func sameSceneInventory(
	left *publicationManifest,
	right *publicationManifest,
) bool {
	if left == nil || right == nil ||
		left.Dataset != right.Dataset ||
		left.TotalSamples != right.TotalSamples ||
		left.ShardCount != right.ShardCount ||
		left.Episodes != right.Episodes ||
		len(left.ShardEntries) != len(right.ShardEntries) {
		return false
	}
	leftNames := make([]string, len(left.ShardEntries))
	rightNames := make([]string, len(right.ShardEntries))
	for index := range left.ShardEntries {
		leftNames[index] = left.ShardEntries[index].Name
		rightNames[index] = right.ShardEntries[index].Name
	}
	sort.Strings(leftNames)
	sort.Strings(rightNames)
	for index := range leftNames {
		if leftNames[index] != rightNames[index] {
			return false
		}
	}
	return true
}
