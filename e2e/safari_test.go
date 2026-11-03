//go:build e2e

package e2e

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/go-wasmcontainers/gpu/internal/webdriver"
)

// runSafari drives the real Safari through safaridriver (the W3C WebDriver
// protocol; Playwright's WebKit is not Safari). The page posts its report to
// the test server, as in the Firefox lane. It needs, once per machine,
//
//	sudo safaridriver --enable
//
// (safaridriver(1)); without it the session is refused and the test FAILS
// with that instruction, it never skips. Safari hosts one WebDriver session
// at a time: do not run two Safari tests in parallel (go test runs the
// tests of a package one after the other).
func runSafari(t *testing.T, url string, reports <-chan string, timeout time.Duration) (string, string) {
	t.Helper()
	driver := os.Getenv("GPU_E2E_EXECUTABLE")
	if driver == "" {
		driver = "/usr/bin/safaridriver"
	}
	ctx := context.Background()
	c, stop, err := webdriver.Start(ctx, driver)
	if err != nil {
		t.Fatal(err)
	}
	defer stop()
	s, err := c.NewSession(ctx, map[string]any{"browserName": "Safari"})
	if err != nil {
		var we *webdriver.Error
		if errors.As(err, &we) && strings.Contains(strings.ToLower(we.Message), "remote automation") {
			t.Fatalf("Safari refused the session: run `sudo safaridriver --enable` once (and tick Develop > Allow Remote Automation): %v", err)
		}
		t.Fatalf("Safari session: %v", err)
	}
	defer s.Delete(ctx)
	version := "Safari " + s.Version
	if err := s.Navigate(ctx, url); err != nil {
		t.Fatalf("%s: navigate: %v", version, err)
	}
	select {
	case r := <-reports:
		return r, version
	case <-time.After(timeout):
		t.Fatalf("%s: no report in %v (no WebGPU in the worker, or the page hung)", version, timeout)
	}
	return "", ""
}
