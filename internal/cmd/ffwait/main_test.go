package main

import (
	"bytes"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"
)

type safeBuf struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (s *safeBuf) Write(p []byte) (int, error) { s.mu.Lock(); defer s.mu.Unlock(); return s.b.Write(p) }
func (s *safeBuf) String() string              { s.mu.Lock(); defer s.mu.Unlock(); return s.b.String() }

func TestNoReportTimesOut(t *testing.T) {
	cmd := RootCmd()
	var errb safeBuf
	cmd.SetErr(&errb)
	cmd.SetOut(&errb)
	cmd.SetArgs([]string{"--no-launch", "--timeout", "200ms"})
	if err := cmd.Execute(); err == nil || !strings.Contains(err.Error(), "no report") {
		t.Fatalf("err = %v", err)
	}
}

func TestServesPageAndCollectsReport(t *testing.T) {
	out := filepath.Join(t.TempDir(), "r.json")
	cmd := RootCmd()
	var errb safeBuf
	cmd.SetErr(&errb)
	cmd.SetOut(&errb)
	cmd.SetArgs([]string{"--no-launch", "--timeout", "10s", "--out", out})
	done := make(chan error, 1)
	go func() { done <- cmd.Execute() }()
	re := regexp.MustCompile(`open (http://localhost:\d+/\S*)`)
	var url string
	for i := 0; i < 200 && url == ""; i++ {
		if m := re.FindStringSubmatch(errb.String()); m != nil {
			url = m[1]
		}
		time.Sleep(10 * time.Millisecond)
	}
	if url == "" {
		t.Fatal("no URL printed")
	}
	resp, err := http.Get(url)
	if err != nil {
		t.Fatal(err)
	}
	var page bytes.Buffer
	page.ReadFrom(resp.Body)
	resp.Body.Close()
	if resp.Header.Get("Cross-Origin-Opener-Policy") != "same-origin" || !strings.Contains(page.String(), "onSubmittedWorkDone") {
		t.Errorf("page headers %v, body %.80s", resp.Header, page.String())
	}
	if _, err := http.Post(strings.Replace(url, "/?", "/report?", 1), "application/json", strings.NewReader(`{"ok":1}`)); err != nil {
		t.Fatal(err)
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(out); string(b) != `{"ok":1}` {
		t.Errorf("report = %q", b)
	}
}
