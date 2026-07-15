package handler

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/go-chi/chi/v5"

	internalauth "github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/auth"
	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/service"
)

type rangeCall struct {
	dataset string
	version string
	shard   string
	offset  int64
	size    int64
}

type fakeDatasetsService struct {
	index       *model.ShardIndex
	indexErr    error
	body        []byte
	buildCalls  []rangeCall
	streamCalls []rangeCall
}

func (*fakeDatasetsService) ListDatasets(context.Context) []model.Dataset {
	panic("unexpected ListDatasets call")
}

func (*fakeDatasetsService) ValidDataset(name string) bool {
	return name == "l2d"
}

func (*fakeDatasetsService) ListDatasetVersions(context.Context, string) ([]model.DatasetVersion, error) {
	panic("unexpected ListDatasetVersions call")
}

func (*fakeDatasetsService) ListShards(context.Context, string, string, int, int) ([]model.Shard, model.Page, error) {
	panic("unexpected ListShards call")
}

func (*fakeDatasetsService) ListSamples(context.Context, string, string, string, int, int) ([]model.Sample, model.Page, error) {
	panic("unexpected ListSamples call")
}

func (*fakeDatasetsService) GetSampleDetail(context.Context, string, string, string, string) (*model.SampleDetail, error) {
	panic("unexpected GetSampleDetail call")
}

func (f *fakeDatasetsService) BuildShardIndex(
	_ context.Context,
	dataset, version, shard string,
) (*model.ShardIndex, error) {
	f.buildCalls = append(f.buildCalls, rangeCall{
		dataset: dataset,
		version: version,
		shard:   shard,
	})
	return f.index, f.indexErr
}

func (f *fakeDatasetsService) StreamTarMemberRange(
	_ context.Context,
	dataset, version, shard string,
	offset, size int64,
) (io.Reader, io.Closer, int64, error) {
	f.streamCalls = append(f.streamCalls, rangeCall{
		dataset: dataset,
		version: version,
		shard:   shard,
		offset:  offset,
		size:    size,
	})
	reader := io.NopCloser(bytes.NewReader(f.body))
	return reader, reader, int64(len(f.body)), nil
}

func newRangeService() *fakeDatasetsService {
	return &fakeDatasetsService{
		index: &model.ShardIndex{
			Version: "v2.1",
			Shard:   "train-000000.tar",
			Samples: []model.IndexSample{{
				Key: "sample",
				Members: map[string]model.MemberRange{
					"cam_0.jpg": {Offset: 512, Size: 4},
				},
			}},
		},
		body: []byte("data"),
	}
}

func requestWithDatasetRoute(target string, params ...string) *http.Request {
	request := httptest.NewRequest(http.MethodGet, target, nil)
	routeContext := chi.NewRouteContext()
	for i := 0; i < len(params); i += 2 {
		routeContext.URLParams.Add(params[i], params[i+1])
	}
	return request.WithContext(context.WithValue(
		request.Context(),
		chi.RouteCtxKey,
		routeContext,
	))
}

func TestValidShardName(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want bool
	}{
		{"plain shard name", "train-000000.tar", true},
		{"shard with underscores", "l2d_shard_000012.tar", true},
		{"traversal with slash", "../secrets.tar", false},
		{"traversal nested", "../../etc/passwd.tar", false},
		{"embedded slash", "a/b.tar", false},
		{"backslash traversal", "..\\secrets.tar", false},
		{"embedded backslash", "a\\b.tar", false},
		{"empty string", "", false},
		{"bare .tar", ".tar", false},
		{"missing .tar suffix", "train-000000", false},
		{"tar in the middle only", "train.tar.gz", false},
		{"dot-dot prefix but valid tar name", "..evil.tar", true}, // no separator: harmless as a single S3 key segment
		{"absolute unix path", "/etc/passwd.tar", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := validShardName(tt.in); got != tt.want {
				t.Errorf("validShardName(%q) = %v, want %v", tt.in, got, tt.want)
			}
		})
	}
}

func TestValidCam(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want bool
	}{
		{"cam_0", "cam_0", true},
		{"cam_6", "cam_6", true},
		{"multi-digit", "cam_12", true},
		{"prefix only", "cam_", false},
		{"non-digit suffix", "cam_x", false},
		{"mixed digit and letter", "cam_1a", false},
		{"empty", "", false},
		{"missing prefix", "0", false},
		{"wrong prefix", "camera_0", false},
		{"traversal in cam", "cam_0/../..", false},
		{"traversal replacing digits", "cam_../x", false},
		{"unicode digit rejected", "cam_٣", false}, // Arabic-Indic three, outside ASCII 0-9
		{"whitespace suffix", "cam_0 ", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := validCam(tt.in); got != tt.want {
				t.Errorf("validCam(%q) = %v, want %v", tt.in, got, tt.want)
			}
		})
	}
}

func TestParsePagination(t *testing.T) {
	tests := []struct {
		name       string
		query      string
		wantLimit  int
		wantOffset int
	}{
		{"defaults when absent", "", defaultLimit, 0},
		{"explicit values", "limit=10&offset=20", 10, 20},
		{"limit clamped to maxLimit", "limit=999999", maxLimit, 0},
		{"limit exactly maxLimit", "limit=1000", maxLimit, 0},
		{"limit just above maxLimit", "limit=1001", maxLimit, 0},
		{"zero limit falls back to default", "limit=0", defaultLimit, 0},
		{"negative limit falls back to default", "limit=-5", defaultLimit, 0},
		{"negative offset falls back to zero", "offset=-1", defaultLimit, 0},
		{"zero offset accepted", "offset=0", defaultLimit, 0},
		{"non-numeric limit ignored", "limit=abc", defaultLimit, 0},
		{"non-numeric offset ignored", "offset=abc", defaultLimit, 0},
		{"float limit ignored", "limit=1.5", defaultLimit, 0},
		{"huge offset passes through", "offset=123456789", defaultLimit, 123456789},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := httptest.NewRequest("GET", "/api/v1/datasets?"+tt.query, nil)
			limit, offset := parsePagination(r)
			if limit != tt.wantLimit {
				t.Errorf("parsePagination(%q) limit = %d, want %d", tt.query, limit, tt.wantLimit)
			}
			if offset != tt.wantOffset {
				t.Errorf("parsePagination(%q) offset = %d, want %d", tt.query, offset, tt.wantOffset)
			}
		})
	}
}

func TestGetImageRequiresValidRangeBeforeS3Access(t *testing.T) {
	handler := NewDatasetsHandler(&service.S3Service{})
	tests := []struct {
		name  string
		query string
	}{
		{name: "range absent"},
		{name: "offset only", query: "?offset=512"},
		{name: "size only", query: "?size=128"},
		{name: "malformed offset", query: "?offset=abc&size=128"},
		{name: "negative offset", query: "?offset=-1&size=128"},
		{name: "zero size", query: "?offset=512&size=0"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			request := httptest.NewRequest(
				"GET",
				"/api/v1/datasets/l2d/shards/train-000000.tar/samples/sample/image/cam_0"+tt.query,
				nil,
			)
			routeContext := chi.NewRouteContext()
			routeContext.URLParams.Add("name", "l2d")
			routeContext.URLParams.Add("shard", "train-000000.tar")
			routeContext.URLParams.Add("key", "sample")
			routeContext.URLParams.Add("cam", "cam_0")
			request = request.WithContext(
				context.WithValue(
					request.Context(),
					chi.RouteCtxKey,
					routeContext,
				),
			)
			response := httptest.NewRecorder()

			handler.GetImage(response, request)

			if response.Code != 400 {
				t.Fatalf("status = %d, want 400", response.Code)
			}
		})
	}
}

func TestGetImageUsesResolvedIndexVersionForRangeRead(t *testing.T) {
	s3 := newRangeService()
	handler := &DatasetsHandler{s3: s3}
	request := requestWithDatasetRoute(
		"/api/v1/datasets/l2d/shards/train-000000.tar/samples/sample/image/cam_0?offset=512&size=4",
		"name", "l2d",
		"shard", "train-000000.tar",
		"key", "sample",
		"cam", "cam_0",
	)
	response := httptest.NewRecorder()

	handler.GetImage(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d: %s", response.Code, http.StatusOK, response.Body.String())
	}
	if len(s3.buildCalls) != 1 || s3.buildCalls[0].version != "" {
		t.Fatalf("BuildShardIndex calls = %+v, want one auto-resolve call", s3.buildCalls)
	}
	if len(s3.streamCalls) != 1 || s3.streamCalls[0].version != "v2.1" {
		t.Fatalf("StreamTarMemberRange calls = %+v, want resolved version v2.1", s3.streamCalls)
	}
}

func TestGetBlobUsesResolvedIndexVersionForRangeRead(t *testing.T) {
	s3 := newRangeService()
	handler := &DatasetsHandler{
		s3:                   s3,
		exactGeoEnabled:      true,
		exactGeoRequiredRole: "exact-geo",
	}
	request := requestWithDatasetRoute(
		"/api/v1/datasets/l2d/shards/train-000000.tar/blob?offset=512&size=4",
		"name", "l2d",
		"shard", "train-000000.tar",
	)
	response := httptest.NewRecorder()

	handler.GetBlob(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d: %s", response.Code, http.StatusOK, response.Body.String())
	}
	if len(s3.buildCalls) != 1 || s3.buildCalls[0].version != "" {
		t.Fatalf("BuildShardIndex calls = %+v, want one auto-resolve call", s3.buildCalls)
	}
	if len(s3.streamCalls) != 1 || s3.streamCalls[0].version != "v2.1" {
		t.Fatalf("StreamTarMemberRange calls = %+v, want resolved version v2.1", s3.streamCalls)
	}
}

func TestShardRangeHandlersMapMissingIndexToNotFound(t *testing.T) {
	tests := []struct {
		name   string
		target string
		params []string
		handle func(*DatasetsHandler, http.ResponseWriter, *http.Request)
	}{
		{
			name:   "image",
			target: "/api/v1/datasets/l2d/shards/train-000000.tar/samples/sample/image/cam_0?offset=512&size=4",
			params: []string{
				"name", "l2d",
				"shard", "train-000000.tar",
				"key", "sample",
				"cam", "cam_0",
			},
			handle: (*DatasetsHandler).GetImage,
		},
		{
			name:   "blob",
			target: "/api/v1/datasets/l2d/shards/train-000000.tar/blob?offset=512&size=4",
			params: []string{
				"name", "l2d",
				"shard", "train-000000.tar",
			},
			handle: (*DatasetsHandler).GetBlob,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s3 := newRangeService()
			s3.indexErr = service.ErrNotFound
			handler := &DatasetsHandler{s3: s3}
			request := requestWithDatasetRoute(tt.target, tt.params...)
			response := httptest.NewRecorder()

			tt.handle(handler, response, request)

			if response.Code != http.StatusNotFound {
				t.Fatalf("status = %d, want %d: %s", response.Code, http.StatusNotFound, response.Body.String())
			}
			if len(s3.streamCalls) != 0 {
				t.Fatalf("range read followed missing index: %+v", s3.streamCalls)
			}
		})
	}
}

func TestShardRangeCacheControlRequiresExplicitVersion(t *testing.T) {
	endpoints := []struct {
		name   string
		target string
		params []string
		handle func(*DatasetsHandler, http.ResponseWriter, *http.Request)
	}{
		{
			name:   "image",
			target: "/api/v1/datasets/l2d/shards/train-000000.tar/samples/sample/image/cam_0?offset=512&size=4",
			params: []string{
				"name", "l2d",
				"shard", "train-000000.tar",
				"key", "sample",
				"cam", "cam_0",
			},
			handle: (*DatasetsHandler).GetImage,
		},
		{
			name:   "blob",
			target: "/api/v1/datasets/l2d/shards/train-000000.tar/blob?offset=512&size=4",
			params: []string{
				"name", "l2d",
				"shard", "train-000000.tar",
			},
			handle: (*DatasetsHandler).GetBlob,
		},
	}
	versions := []struct {
		name  string
		query string
		want  string
	}{
		{name: "auto-resolved", want: "no-store"},
		{name: "explicit", query: "&version=v2.1", want: "public, max-age=3600"},
	}
	for _, endpoint := range endpoints {
		for _, version := range versions {
			t.Run(endpoint.name+"/"+version.name, func(t *testing.T) {
				s3 := newRangeService()
				handler := &DatasetsHandler{s3: s3}
				request := requestWithDatasetRoute(
					endpoint.target+version.query,
					endpoint.params...,
				)
				response := httptest.NewRecorder()

				endpoint.handle(handler, response, request)

				if response.Code != http.StatusOK {
					t.Fatalf("status = %d, want %d: %s", response.Code, http.StatusOK, response.Body.String())
				}
				if got := response.Header().Get("Cache-Control"); got != version.want {
					t.Fatalf("Cache-Control = %q, want %q", got, version.want)
				}
			})
		}
	}
}

func TestIndexWithoutExactGeoDoesNotMutateCachedIndex(t *testing.T) {
	pose := &model.GeoPose{
		LatitudeDeg:           35.6812,
		LongitudeDeg:          139.7671,
		HeadingDegCWFromNorth: 90,
		TimestampNS:           42,
	}
	cached := &model.ShardIndex{
		Fps:     10,
		Version: "v2.1",
		Shard:   "train-000000.tar",
		Samples: []model.IndexSample{{
			Key:         "l2d-v1-e000001-f000042",
			SampleUID:   "l2d-v1-e000001-f000042",
			PoseCurrent: pose,
			Members: map[string]model.MemberRange{
				"cam_0.jpg": {Offset: 512, Size: 128},
				"pose.npy":  {Offset: 1024, Size: 36},
				"gps.npy":   {Offset: 1536, Size: 1040},
			},
		}},
	}

	redacted := indexWithoutExactGeo(cached)
	if redacted == cached {
		t.Fatal("redaction returned the cached index pointer")
	}
	if redacted.Samples[0].PoseCurrent != nil {
		t.Fatal("redacted index still contains exact pose")
	}
	if cached.Samples[0].PoseCurrent != pose {
		t.Fatal("redaction mutated the cached index")
	}
	if redacted.Samples[0].SampleUID != cached.Samples[0].SampleUID {
		t.Fatal("redaction changed non-sensitive sample fields")
	}
	if _, ok := redacted.Samples[0].Members["pose.npy"]; ok {
		t.Fatal("redacted index still contains pose member offset")
	}
	if _, ok := redacted.Samples[0].Members["gps.npy"]; ok {
		t.Fatal("redacted index still contains GPS member offset")
	}
	if _, ok := redacted.Samples[0].Members["cam_0.jpg"]; !ok {
		t.Fatal("redaction removed camera member")
	}
	if len(cached.Samples[0].Members) != 3 {
		t.Fatal("redaction mutated cached member map")
	}
}

func TestExactGeoResponsesAreNotCacheable(t *testing.T) {
	pose := &model.GeoPose{LatitudeDeg: 35, LongitudeDeg: 139}
	s3 := newRangeService()
	s3.index.Samples[0].PoseCurrent = pose
	s3.index.Samples[0].Members["pose.npy"] = model.MemberRange{
		Offset: 1024,
		Size:   4,
	}
	handler := &DatasetsHandler{
		s3:                   s3,
		exactGeoEnabled:      true,
		exactGeoRequiredRole: "exact-geo",
	}
	withPrincipal := func(request *http.Request) *http.Request {
		return request.WithContext(internalauth.WithPrincipal(
			request.Context(),
			internalauth.Principal{
				Subject: "user-1",
				Roles:   []string{"exact-geo"},
			},
		))
	}

	indexRequest := withPrincipal(requestWithDatasetRoute(
		"/api/v1/datasets/l2d/shards/train-000000.tar/index?version=v2.1",
		"name", "l2d",
		"shard", "train-000000.tar",
	))
	indexResponse := httptest.NewRecorder()
	handler.GetShardIndex(indexResponse, indexRequest)
	if indexResponse.Code != http.StatusOK {
		t.Fatalf("index status = %d: %s", indexResponse.Code, indexResponse.Body.String())
	}
	if got := indexResponse.Header().Get("Cache-Control"); got != "private, no-store" {
		t.Fatalf("index Cache-Control = %q", got)
	}

	blobRequest := withPrincipal(requestWithDatasetRoute(
		"/api/v1/datasets/l2d/shards/train-000000.tar/blob?offset=1024&size=4&version=v2.1",
		"name", "l2d",
		"shard", "train-000000.tar",
	))
	blobResponse := httptest.NewRecorder()
	handler.GetBlob(blobResponse, blobRequest)
	if blobResponse.Code != http.StatusOK {
		t.Fatalf("blob status = %d: %s", blobResponse.Code, blobResponse.Body.String())
	}
	if got := blobResponse.Header().Get("Cache-Control"); got != "private, no-store" {
		t.Fatalf("blob Cache-Control = %q", got)
	}
}

func TestCameraMemberRangeRequiresMatchingSampleAndCamera(t *testing.T) {
	index := &model.ShardIndex{Samples: []model.IndexSample{{
		Key: "sample-a",
		Members: map[string]model.MemberRange{
			"cam_0.jpg": {Offset: 512, Size: 128},
		},
	}}}
	got, ok := cameraMemberRange(index, "sample-a", "cam_0.jpg")
	if !ok || got.Offset != 512 || got.Size != 128 {
		t.Fatalf("camera range = %+v, %v", got, ok)
	}
	if _, ok := cameraMemberRange(index, "sample-b", "cam_0.jpg"); ok {
		t.Fatal("matched a range from another sample")
	}
	if _, ok := cameraMemberRange(index, "sample-a", "cam_1.jpg"); ok {
		t.Fatal("matched a range from another camera")
	}
}

func TestRangeOverlapsExactGeo(t *testing.T) {
	index := &model.ShardIndex{Samples: []model.IndexSample{{
		Members: map[string]model.MemberRange{
			"cam_0.jpg": {Offset: 512, Size: 128},
			"pose.npy":  {Offset: 1024, Size: 36},
			"gps.npy":   {Offset: 1536, Size: 1040},
		},
	}}}
	tests := []struct {
		name   string
		offset int64
		size   int64
		want   bool
	}{
		{"camera only", 512, 128, false},
		{"ends at pose", 896, 128, false},
		{"starts at pose", 1024, 1, true},
		{"spans pose", 900, 200, true},
		{"starts at GPS", 1536, 1, true},
		{"after GPS", 2576, 10, false},
		{"invalid size", 0, 0, true},
		{"overflow", int64(^uint64(0) >> 1), 2, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := rangeOverlapsExactGeo(index, tt.offset, tt.size); got != tt.want {
				t.Fatalf("rangeOverlapsExactGeo(%d, %d) = %v, want %v", tt.offset, tt.size, got, tt.want)
			}
		})
	}
}
