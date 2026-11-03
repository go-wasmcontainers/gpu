package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestParseLoad(t *testing.T) {
	got, err := ParseLoad("18:22  up 1 day,  7:51, 1 user, load averages: 3.39 3.42 4.19")
	if err != nil || got != 3.39 {
		t.Fatalf("got %v, %v", got, err)
	}
	if _, err := ParseLoad("nothing"); err == nil {
		t.Fatal("no load average must be an error")
	}
}

func TestPercentile(t *testing.T) {
	xs := []float64{5, 1, 3, 2, 4}
	for p, want := range map[float64]float64{0: 1, 50: 3, 100: 5, 25: 2} {
		if got := Percentile(xs, p); got != want {
			t.Errorf("p%v = %v, want %v", p, got, want)
		}
	}
	if got := Percentile([]float64{1, 2}, 50); got != 1.5 {
		t.Errorf("interpolation: %v", got)
	}
	if !math.IsNaN(Percentile(nil, 50)) {
		t.Error("empty input must be NaN")
	}
}

func TestParseBench(t *testing.T) {
	pp, ppN, tg, tgN, err := ParseBench([]byte(`[{"n_prompt":128,"n_gen":0,"avg_ts":210.5},{"n_prompt":0,"n_gen":64,"avg_ts":55.25}]`))
	if err != nil || pp != 210.5 || ppN != 128 || tg != 55.25 || tgN != 64 {
		t.Fatalf("%v %v %v %v %v", pp, ppN, tg, tgN, err)
	}
	if _, _, _, _, err := ParseBench([]byte(`[{"n_prompt":128,"n_gen":0,"avg_ts":1}]`)); err == nil {
		t.Fatal("a missing tg row must be an error")
	}
}

func TestCheckAdapter(t *testing.T) {
	if err := CheckAdapter("apple", "metal-3", false); err != nil {
		t.Error(err)
	}
	for _, c := range [][2]string{{"google", "swiftshader"}, {"apple", "swiftshader"}, {"nvidia", "ada"}, {"", ""}} {
		if CheckAdapter(c[0], c[1], false) == nil {
			t.Errorf("%v accepted", c)
		}
	}
	if CheckAdapter("apple", "metal-3", true) == nil {
		t.Error("fallback accepted")
	}
}

func TestInterleaveRotatesAndDefersOnLoad(t *testing.T) {
	var order []string
	load := []float64{1, 9, 1, 1, 1, 1, 1, 1}
	i := 0
	var out bytes.Buffer
	e := &Env{MaxLoad: 8, MaxRedo: 3, Wait: time.Second, Out: &out,
		loadFn: func(context.Context) (float64, error) { l := load[min(i, len(load)-1)]; i++; return l, nil },
		sleep:  func(time.Duration) {},
		runLlama: func(_ context.Context, _ *Env, c Config) (Sample, error) {
			order = append(order, c.Name)
			return Sample{PP: 1, TG: 1}, nil
		}}
	e.runWebLLM = e.runLlama
	s, err := Interleave(context.Background(), e, []Config{{Name: "a"}, {Name: "b"}}, 2)
	if err != nil || len(s) != 4 {
		t.Fatalf("%v %d", err, len(s))
	}
	if strings.Join(order, "") != "abba" {
		t.Errorf("order %v", order)
	}
	if !strings.Contains(out.String(), "waiting") {
		t.Error("a load above the limit must make the round wait")
	}
	e.loadFn = func(context.Context) (float64, error) { return 99, nil }
	if _, err := Interleave(context.Background(), e, []Config{{Name: "a"}}, 1); err == nil || !strings.Contains(err.Error(), "deferred") {
		t.Errorf("a load that never drops must defer: %v", err)
	}
}

func TestFetchHFVerifiesAndRefusesCorruption(t *testing.T) {
	body := []byte("GGUF-weights")
	sum := sha256.Sum256(body)
	oid := hex.EncodeToString(sum[:])
	commit := strings.Repeat("a", 40)
	corrupt := false
	mux := http.NewServeMux()
	mux.HandleFunc("/api/models/o/m", func(w http.ResponseWriter, r *http.Request) { fmt.Fprintf(w, `{"sha":%q}`, commit) })
	mux.HandleFunc("/api/models/o/m/tree/"+commit, func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `[{"type":"file","oid":"x","size":%d,"path":"w.gguf","lfs":{"oid":%q,"size":%d}},{"type":"file","oid":"y","size":1,"path":"../evil"}]`, len(body), oid, len(body))
	})
	mux.HandleFunc("/o/m/resolve/"+commit+"/w.gguf", func(w http.ResponseWriter, r *http.Request) {
		if corrupt {
			w.Write([]byte("GGUF-weighTs"))
			return
		}
		w.Write(body)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()
	old := hfBase
	hfBase = srv.URL
	defer func() { hfBase = old }()
	dir := t.TempDir()

	pins, err := FetchHF(context.Background(), dir, "o/m", []string{"w.gguf"})
	if err != nil || len(pins) != 1 || pins[0].SHA256 != oid || pins[0].Commit != commit || pins[0].Verified != "lfs-sha256" {
		t.Fatalf("%+v %v", pins, err)
	}
	// Corrupt body: refused, nothing left at the destination.
	os.Remove(filepath.Join(dir, "o", "m", "w.gguf"))
	corrupt = true
	if _, err := FetchHF(context.Background(), dir, "o/m", []string{"w.gguf"}); err == nil {
		t.Fatal("a body that does not match the Hub's sha256 must be refused")
	}
	if _, err := os.Stat(filepath.Join(dir, "o", "m", "w.gguf")); err == nil {
		t.Fatal("a refused file must not be left in place")
	}
	// A path escaping the corpus is refused (listing contains "../evil").
	if _, err := FetchHF(context.Background(), dir, "o/m", nil); err == nil {
		t.Fatal("unsafe path accepted")
	}
	if _, err := FetchHF(context.Background(), dir, "../x", nil); err == nil {
		t.Fatal("bad repo accepted")
	}
}

func TestServeRefusesEscape(t *testing.T) {
	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, "mlc-ai", "m"), 0o755)
	os.WriteFile(filepath.Join(dir, "mlc-ai", "m", "f.json"), []byte("{}"), 0o644)
	os.WriteFile(filepath.Join(dir, "secret"), []byte("s"), 0o644)
	addr, stop, err := serve(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer stop()
	get := func(p string) int {
		resp, err := http.Get(addr + p)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		return resp.StatusCode
	}
	if get("/mlc/m/resolve/main/f.json") != 200 {
		t.Error("model file not served")
	}
	if get("/mlc/../secret") == 200 || get("/mlc/%2e%2e/secret") == 200 {
		t.Error("path escape served")
	}
	if get("/other") != 404 {
		t.Error("unknown path served")
	}
}

func TestRootCmdRefusesFewRounds(t *testing.T) {
	c := RootCmd()
	var out bytes.Buffer
	c.SetOut(&out)
	c.SetErr(&out)
	c.SetArgs([]string{"run", "--corpus", "x", "--smollm2", "x", "--qwen", "x", "--out", "x", "--rounds", "3"})
	if err := c.Execute(); err == nil || !strings.Contains(err.Error(), "at least 5") {
		t.Fatalf("%v", err)
	}
}

func TestConfigsDefaultToTheBottle(t *testing.T) {
	cs := configs("c", "s.gguf", "q.gguf", "/bottle/llama-bench", "", "", false)
	if len(cs) != 4 {
		t.Fatalf("%d rows without a hand build and without WebLLM", len(cs))
	}
	for _, c := range cs {
		if c.Bin != "/bottle/llama-bench" || !strings.HasSuffix(c.Name, "[bottle]") {
			t.Errorf("%+v", c)
		}
	}
	cs = configs("c", "s.gguf", "q.gguf", "/bottle/llama-bench", "/hand/cpu", "/hand/metal", false)
	hand := 0
	for _, c := range cs {
		if strings.HasSuffix(c.Name, "[hand]") {
			hand++
			if c.Bin == "/bottle/llama-bench" {
				t.Error("a hand row runs the bottle")
			}
		}
	}
	if hand != 4 {
		t.Errorf("%d hand rows", hand)
	}
}

func TestBenchArgs(t *testing.T) {
	cpu := strings.Join(benchArgs(Config{Kind: "llamacpp-cpu", GGUF: "m"}, 128, 128), " ")
	for _, want := range []string{"-t 1", "-ngl 0", "-dev none", "-nopo 1", "-p 128", "-n 128"} {
		if !strings.Contains(cpu, want) {
			t.Errorf("cpu args %q lack %q", cpu, want)
		}
	}
	if m := strings.Join(benchArgs(Config{Kind: "llamacpp-metal", GGUF: "m"}, 128, 128), " "); !strings.Contains(m, "-ngl 99") || strings.Contains(m, "-dev none") {
		t.Errorf("metal args %q", m)
	}
}

func TestCheckNoGPU(t *testing.T) {
	ok := "load_tensors: offloaded 0/25 layers to GPU\nload_tensors:   CPU_Mapped model buffer size = 73 MiB"
	if err := CheckNoGPU(ok); err != nil {
		t.Error(err)
	}
	if err := CheckNoGPU("load_tensors:   CPU_Mapped model buffer size = 73 MiB"); err != nil {
		t.Errorf("a build without a GPU backend: %v", err)
	}
	for _, bad := range []string{"load_tensors: offloaded 25/25 layers to GPU", ok + "\nload_tensors:   MTL0_Mapped model buffer size = 1 MiB", "nothing"} {
		if CheckNoGPU(bad) == nil {
			t.Errorf("accepted %q", bad)
		}
	}
}
