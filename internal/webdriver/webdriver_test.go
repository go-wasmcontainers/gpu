package webdriver

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// fake is a minimal W3C WebDriver server.
func fake(t *testing.T, got *[]string) *Client {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		*got = append(*got, r.Method+" "+r.URL.Path+" "+string(b))
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.URL.Path == "/session" && strings.Contains(string(b), `"Chrome"`):
			w.WriteHeader(500)
			io.WriteString(w, `{"value":{"error":"session not created","message":"Could not create a session: You must enable 'Allow Remote Automation'","stacktrace":""}}`)
		case r.URL.Path == "/session":
			io.WriteString(w, `{"value":{"sessionId":"S1","capabilities":{"browserName":"Safari","browserVersion":"27.2"}}}`)
		case r.URL.Path == "/session/S1/execute/sync":
			io.WriteString(w, `{"value":"ok"}`)
		default:
			io.WriteString(w, `{"value":null}`)
		}
	}))
	t.Cleanup(srv.Close)
	return &Client{Base: srv.URL}
}

func TestSessionLifecycle(t *testing.T) {
	var got []string
	c := fake(t, &got)
	ctx := context.Background()
	s, err := c.NewSession(ctx, map[string]any{"browserName": "Safari"})
	if err != nil {
		t.Fatal(err)
	}
	if s.ID != "S1" || s.Version != "27.2" {
		t.Fatalf("session %+v", s)
	}
	if err := s.Navigate(ctx, "http://localhost:1/index.html"); err != nil {
		t.Fatal(err)
	}
	v, err := s.Execute(ctx, "return 1")
	if err != nil || string(v) != `"ok"` {
		t.Fatalf("execute: %s, %v", v, err)
	}
	if err := s.Delete(ctx); err != nil {
		t.Fatal(err)
	}
	want := []string{`POST /session {"capabilities":{"alwaysMatch":{"browserName":"Safari"}}}`,
		`POST /session/S1/url {"url":"http://localhost:1/index.html"}`,
		`POST /session/S1/execute/sync {"args":[],"script":"return 1"}`,
		`DELETE /session/S1 `}
	if len(got) != len(want) {
		t.Fatalf("requests %q", got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("request %d: %q, want %q", i, got[i], want[i])
		}
	}
}

func TestErrorResponse(t *testing.T) {
	var got []string
	c := fake(t, &got)
	_, err := c.NewSession(context.Background(), map[string]any{"browserName": "Chrome"})
	var e *Error
	if !errors.As(err, &e) || e.Status != 500 || e.Code != "session not created" || !strings.Contains(e.Message, "Allow Remote Automation") {
		t.Fatalf("error %v", err)
	}
}

func TestNotJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, "<html>") }))
	defer srv.Close()
	c := &Client{Base: srv.URL}
	if _, err := c.NewSession(context.Background(), nil); err == nil {
		t.Fatal("HTML answer accepted")
	}
}

func TestOversizedResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := json.Marshal(map[string]string{"value": strings.Repeat("x", maxResponse+10)})
		w.Write(b)
	}))
	defer srv.Close()
	c := &Client{Base: srv.URL}
	if _, err := c.NewSession(context.Background(), nil); err == nil || !strings.Contains(err.Error(), "too large") {
		t.Fatalf("err %v", err)
	}
}

func FuzzDecode(f *testing.F) {
	f.Add(`{"value":{"sessionId":"a","capabilities":{}}}`)
	f.Add(`{"value":{"error":"x","message":"y"}}`)
	f.Add(`not json`)
	f.Fuzz(func(t *testing.T, body string) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, body) }))
		defer srv.Close()
		c := &Client{Base: srv.URL}
		c.NewSession(context.Background(), nil) // must not panic
	})
}
