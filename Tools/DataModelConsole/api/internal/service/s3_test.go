package service

import (
	"archive/tar"
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
)

func TestSampleKeyOf(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{
			name: "normal webdataset member",
			in:   "ep0_000064.cam_0.jpg",
			want: "ep0_000064",
		},
		{
			name: "single extension",
			in:   "ep0_000064.json",
			want: "ep0_000064",
		},
		{
			name: "no dot returns whole base",
			in:   "README",
			want: "README",
		},
		{
			name: "hidden file (leading dot) is not truncated to empty key",
			in:   ".hidden",
			want: ".hidden",
		},
		{
			name: "hidden file with extension keeps full base (i==0 guard)",
			in:   ".hidden.json",
			want: ".hidden.json",
		},
		{
			name: "nested path uses base name",
			in:   "a/b/ep1_000001.cam_2.jpg",
			want: "ep1_000001",
		},
		{
			name: "trailing dot",
			in:   "name.",
			want: "name",
		},
		{
			name: "empty name degrades to path.Base dot",
			in:   "",
			want: ".",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := sampleKeyOf(tt.in); got != tt.want {
				t.Errorf("sampleKeyOf(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestPaginate(t *testing.T) {
	items := func(n int) []int {
		out := make([]int, n)
		for i := range out {
			out[i] = i
		}
		return out
	}

	tests := []struct {
		name       string
		total      int
		limit      int
		offset     int
		wantItems  []int
		wantLimit  int
		wantOffset int
		wantMore   bool
	}{
		{
			name:       "offset beyond total returns empty page",
			total:      5,
			limit:      10,
			offset:     100,
			wantItems:  []int{},
			wantLimit:  10,
			wantOffset: 5, // clamped to total
			wantMore:   false,
		},
		{
			name:       "zero limit falls back to default 50",
			total:      5,
			limit:      0,
			offset:     0,
			wantItems:  []int{0, 1, 2, 3, 4},
			wantLimit:  50,
			wantOffset: 0,
			wantMore:   false,
		},
		{
			name:       "negative limit falls back to default 50",
			total:      3,
			limit:      -1,
			offset:     0,
			wantItems:  []int{0, 1, 2},
			wantLimit:  50,
			wantOffset: 0,
			wantMore:   false,
		},
		{
			name:       "exact boundary: last full page has More=false",
			total:      10,
			limit:      5,
			offset:     5,
			wantItems:  []int{5, 6, 7, 8, 9},
			wantLimit:  5,
			wantOffset: 5,
			wantMore:   false,
		},
		{
			name:       "first page of two has More=true",
			total:      10,
			limit:      5,
			offset:     0,
			wantItems:  []int{0, 1, 2, 3, 4},
			wantLimit:  5,
			wantOffset: 0,
			wantMore:   true,
		},
		{
			name:       "partial trailing page",
			total:      7,
			limit:      5,
			offset:     5,
			wantItems:  []int{5, 6},
			wantLimit:  5,
			wantOffset: 5,
			wantMore:   false,
		},
		{
			name:       "negative offset clamped to zero",
			total:      4,
			limit:      2,
			offset:     -3,
			wantItems:  []int{0, 1},
			wantLimit:  2,
			wantOffset: 0,
			wantMore:   true,
		},
		{
			name:       "offset equal to total returns empty, More=false",
			total:      4,
			limit:      2,
			offset:     4,
			wantItems:  []int{},
			wantLimit:  2,
			wantOffset: 4,
			wantMore:   false,
		},
		{
			name:       "empty input",
			total:      0,
			limit:      10,
			offset:     0,
			wantItems:  []int{},
			wantLimit:  10,
			wantOffset: 0,
			wantMore:   false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			in := items(tt.total)
			got, page := paginate(in, tt.limit, tt.offset, tt.total)

			if len(got) != len(tt.wantItems) {
				t.Fatalf("paginate items len = %d, want %d (got %v)", len(got), len(tt.wantItems), got)
			}
			for i := range got {
				if got[i] != tt.wantItems[i] {
					t.Fatalf("paginate items = %v, want %v", got, tt.wantItems)
				}
			}
			if page.Limit != tt.wantLimit {
				t.Errorf("page.Limit = %d, want %d", page.Limit, tt.wantLimit)
			}
			if page.Offset != tt.wantOffset {
				t.Errorf("page.Offset = %d, want %d", page.Offset, tt.wantOffset)
			}
			if page.Total != tt.total {
				t.Errorf("page.Total = %d, want %d", page.Total, tt.total)
			}
			if page.More != tt.wantMore {
				t.Errorf("page.More = %v, want %v", page.More, tt.wantMore)
			}
		})
	}
}

// TestPaginate_HugeOffsetDoesNotPanic is a regression test for an integer
// overflow: paginate computes end := offset + limit BEFORE clamping offset to
// total, so an offset near MaxInt overflows end to a negative value and
// items[offset:end] panics with "slice bounds out of range". The offset is
// remotely attacker-controlled: handler.parsePagination accepts any
// non-negative int (?offset=9223372036854775807 parses fine on 64-bit), so
// GET /api/v1/datasets/l2d/shards?offset=9223372036854775807 panics per
// request (recovered by chi middleware.Recoverer into a 500, with a stack
// trace logged). Correct behavior: return an empty page with More=false.
// This test FAILS until the production fix lands (clamp offset before
// computing end, and guard end < offset for overflow).
func TestPaginate_HugeOffsetDoesNotPanic(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("paginate(items, 50, math.MaxInt, 5) panicked: %v (int overflow in end := offset + limit)", r)
		}
	}()
	items := []int{0, 1, 2, 3, 4}
	got, page := paginate(items, 50, math.MaxInt, len(items))
	if len(got) != 0 {
		t.Errorf("expected empty page for huge offset, got %v", got)
	}
	if page.More {
		t.Errorf("page.More = true, want false for offset beyond total")
	}
}

func TestCountingReader_AccumulatesAcrossReads(t *testing.T) {
	src := strings.Repeat("x", 1000)
	cr := &countingReader{r: strings.NewReader(src)}

	buf := make([]byte, 137) // deliberately not a divisor of 1000
	var total int64
	for {
		n, err := cr.Read(buf)
		total += int64(n)
		if cr.n != total {
			t.Fatalf("countingReader.n = %d after reading %d bytes total", cr.n, total)
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("unexpected read error: %v", err)
		}
	}
	if total != int64(len(src)) {
		t.Fatalf("read %d bytes, want %d", total, len(src))
	}
	if cr.n != int64(len(src)) {
		t.Fatalf("countingReader.n = %d, want %d", cr.n, len(src))
	}

	// EOF reads after exhaustion must not change the count.
	if n, err := cr.Read(buf); n != 0 || err != io.EOF {
		t.Fatalf("post-EOF read = (%d, %v), want (0, EOF)", n, err)
	}
	if cr.n != int64(len(src)) {
		t.Fatalf("countingReader.n changed after EOF read: %d", cr.n)
	}
}

func TestFullTarScanSemaphoreBoundsProcessWide(t *testing.T) {
	acquired := make(chan func(), maxConcurrentFullTarScans+1)
	errs := make(chan error, maxConcurrentFullTarScans+1)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	for range maxConcurrentFullTarScans + 1 {
		go func() {
			release, err := acquireFullTarScan(ctx)
			if err != nil {
				errs <- err
				return
			}
			acquired <- release
		}()
	}

	releases := make([]func(), 0, maxConcurrentFullTarScans+1)
	for range maxConcurrentFullTarScans {
		select {
		case release := <-acquired:
			releases = append(releases, release)
		case err := <-errs:
			t.Fatalf("acquire index-build slot: %v", err)
		case <-time.After(time.Second):
			t.Fatal("timed out acquiring allowed index-build slots")
		}
	}

	select {
	case release := <-acquired:
		release()
		t.Fatalf(
			"more than %d full-tar scans acquired concurrently",
			maxConcurrentFullTarScans,
		)
	case err := <-errs:
		t.Fatalf("unexpected acquire error: %v", err)
	case <-time.After(50 * time.Millisecond):
	}

	releases[0]()
	select {
	case release := <-acquired:
		releases = append(releases, release)
	case err := <-errs:
		t.Fatalf("queued acquire failed after release: %v", err)
	case <-time.After(time.Second):
		t.Fatal("queued shard build did not acquire the released slot")
	}
	for _, release := range releases[1:] {
		release()
	}
}

func TestBuildShardIndexWaiterHonorsContextCancellation(t *testing.T) {
	service, _ := newPublicationTestService(t)
	const shard = "scene-a-train-000000.tar"
	service.indexSF["kitscenes/v2.1/"+shard] = &shardIndexBuild{
		done: make(chan struct{}),
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := service.BuildShardIndex(
		ctx,
		"kitscenes",
		"v2.1",
		shard,
	)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("BuildShardIndex error = %v, want context.Canceled", err)
	}
}

type cachedShardIndexStore struct {
	consoleStore
	index    *model.ShardIndex
	getCalls int
}

func (s *cachedShardIndexStore) GetShardIndex(
	context.Context,
	string,
	string,
	string,
) (*model.ShardIndex, error) {
	s.getCalls++
	return s.index, nil
}

func TestListSamplesUsesCachedShardIndex(t *testing.T) {
	service, client := newPublicationTestService(t)
	const (
		shard     = "scene-a-train-000000.tar"
		sampleUID = "kitscenes-v1-scene-a-f000000"
	)
	cache := &cachedShardIndexStore{
		index: &model.ShardIndex{
			Fps:     indexFps,
			Version: "v2.1",
			Shard:   shard,
			Samples: []model.IndexSample{{
				Key:       sampleUID,
				SampleUID: sampleUID,
				Members: map[string]model.MemberRange{
					"meta.json": {Offset: 1024, Size: 20},
					"cam_0.jpg": {Offset: 512, Size: 4},
				},
			}},
		},
	}
	service.store = cache

	samples, page, err := service.ListSamples(
		context.Background(),
		"kitscenes",
		"v2.1",
		shard,
		50,
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	if cache.getCalls != 1 {
		t.Fatalf("shard index reads = %d, want 1", cache.getCalls)
	}
	shardKey := "kitscenes/v2.1/shards/" + shard
	if client.getCalls[shardKey] != 0 {
		t.Fatalf("cached sample list fetched shard %d times", client.getCalls[shardKey])
	}
	if page.Total != 1 || len(samples) != 1 {
		t.Fatalf("samples/page = %+v / %+v", samples, page)
	}
	if len(samples[0].Members) != 2 ||
		samples[0].Members[0].Name != sampleUID+".cam_0.jpg" ||
		samples[0].Members[1].Name != sampleUID+".meta.json" {
		t.Fatalf("member order/shape changed: %+v", samples[0].Members)
	}
}

func TestSampleTarScansUseSharedProcessSemaphore(t *testing.T) {
	service, client := newPublicationTestService(t)
	if _, err := service.loadPublicationManifest(
		context.Background(), "kitscenes", "v2.1",
	); err != nil {
		t.Fatal(err)
	}

	releases := make([]func(), 0, maxConcurrentFullTarScans)
	for range maxConcurrentFullTarScans {
		release, err := acquireFullTarScan(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		releases = append(releases, release)
	}
	defer func() {
		for _, release := range releases {
			release()
		}
	}()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	const shard = "scene-a-train-000000.tar"
	shardKey := "kitscenes/v2.1/shards/" + shard
	before := client.getCalls[shardKey]
	if _, _, err := service.ListSamples(
		ctx, "kitscenes", "v2.1", shard, 50, 0,
	); !errors.Is(err, context.Canceled) {
		t.Fatalf("ListSamples saturated scan error = %v", err)
	}
	if _, err := service.GetSampleDetail(
		ctx,
		"kitscenes",
		"v2.1",
		shard,
		"kitscenes-v1-scene-a-f000000",
	); !errors.Is(err, context.Canceled) {
		t.Fatalf("GetSampleDetail saturated scan error = %v", err)
	}
	if client.getCalls[shardKey] != before {
		t.Fatal("sample scan reached S3 without acquiring the shared slot")
	}
}

func TestDecodeEgoPayloadRejectsMalformedData(t *testing.T) {
	valid := make([]byte, egoPayloadBytes)
	if got, err := decodeEgoPayload(valid); err != nil ||
		len(got) != egoTotalFloats {
		t.Fatalf("valid ego payload = %d floats, %v", len(got), err)
	}

	withFloat := func(index int, value float32) []byte {
		body := append([]byte(nil), valid...)
		binary.LittleEndian.PutUint32(
			body[index*4:],
			math.Float32bits(value),
		)
		return body
	}
	tests := []struct {
		name string
		body []byte
	}{
		{name: "short", body: valid[:len(valid)-1]},
		{name: "trailing byte", body: append(append([]byte(nil), valid...), 0)},
		{name: "nan history", body: withFloat(0, float32(math.NaN()))},
		{name: "positive infinity future", body: withFloat(300, float32(math.Inf(1)))},
		{name: "negative infinity future", body: withFloat(383, float32(math.Inf(-1)))},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := decodeEgoPayload(test.body); err == nil {
				t.Fatal("malformed ego payload was accepted")
			}
		})
	}
}

func TestGetSampleDetailRejectsNonFiniteEgoPayload(t *testing.T) {
	service, client := newPublicationTestService(t)
	if _, err := service.loadPublicationManifest(
		context.Background(), "kitscenes", "v2.1",
	); err != nil {
		t.Fatal(err)
	}
	const (
		shard     = "scene-a-train-000000.tar"
		sampleUID = "kitscenes-v1-scene-a-f000000"
	)
	ego := make([]byte, egoPayloadBytes)
	binary.LittleEndian.PutUint32(
		ego[(egoTotalFloats-1)*4:],
		math.Float32bits(float32(math.NaN())),
	)
	shardKey := "kitscenes/v2.1/shards/" + shard
	object := client.objects[shardKey]
	object.body = encodeTestTar(t, []testTarMember{{
		name: sampleUID + ".ego.npy",
		body: ego,
	}})
	client.objects[shardKey] = object

	if _, err := service.GetSampleDetail(
		context.Background(),
		"kitscenes",
		"v2.1",
		shard,
		sampleUID,
	); err == nil || !strings.Contains(err.Error(), "non-finite") {
		t.Fatalf("non-finite detail payload error = %v", err)
	}
}

// TestCountingReader_TarDataOffsets validates the invariant ListSamples relies
// on: after tar.Reader.Next() the counting reader's n is exactly the byte
// offset of the member's data within the tar stream, so raw[n:n+size]
// reproduces the member content (the basis for Phase 2 range-GET extraction).
func TestCountingReader_TarDataOffsets(t *testing.T) {
	files := []struct {
		name string
		body string
	}{
		{"ep0_000000.cam_0.jpg", strings.Repeat("A", 700)}, // spans >1 block, forces padding
		{"ep0_000000.json", `{"speed": 1.0}`},
		{"ep0_000001.cam_0.jpg", strings.Repeat("B", 512)}, // exact block size
	}

	var raw bytes.Buffer
	tw := tar.NewWriter(&raw)
	for _, f := range files {
		if err := tw.WriteHeader(&tar.Header{
			Name:     f.name,
			Mode:     0o644,
			Size:     int64(len(f.body)),
			Typeflag: tar.TypeReg,
		}); err != nil {
			t.Fatalf("write header %s: %v", f.name, err)
		}
		if _, err := tw.Write([]byte(f.body)); err != nil {
			t.Fatalf("write body %s: %v", f.name, err)
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatalf("close tar writer: %v", err)
	}
	rawBytes := raw.Bytes()

	cr := &countingReader{r: bytes.NewReader(rawBytes)}
	tr := tar.NewReader(cr)
	i := 0
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("tar next: %v", err)
		}
		if hdr.Typeflag != tar.TypeReg {
			continue
		}
		if i >= len(files) {
			t.Fatalf("unexpected extra member %q", hdr.Name)
		}
		offset := cr.n // same accounting as ListSamples
		want := files[i].body
		if hdr.Size != int64(len(want)) {
			t.Errorf("member %s size = %d, want %d", hdr.Name, hdr.Size, len(want))
		}
		got := string(rawBytes[offset : offset+hdr.Size])
		if got != want {
			t.Errorf("member %s: raw[%d:%d] does not match member content", hdr.Name, offset, offset+hdr.Size)
		}
		i++
	}
	if i != len(files) {
		t.Fatalf("iterated %d members, want %d", i, len(files))
	}
}
