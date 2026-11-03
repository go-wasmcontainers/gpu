// Package webdriver is a small W3C WebDriver client (https://www.w3.org/TR/webdriver/):
// just what the e2e lane for Safari needs, to drive the real Safari through
// safaridriver (Playwright's WebKit is not Safari). It starts the driver,
// creates a session, navigates and deletes the session.
package webdriver

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os/exec"
	"time"
)

// maxResponse bounds what a driver may answer.
const maxResponse = 1 << 20

// Client talks to one WebDriver server.
type Client struct {
	Base string // http://127.0.0.1:port
	HTTP *http.Client
}

// Error is a WebDriver error response (W3C section 6.6).
type Error struct {
	Status  int
	Code    string // "error" field, e.g. "session not created"
	Message string
}

func (e *Error) Error() string {
	return fmt.Sprintf("webdriver: %d %s: %s", e.Status, e.Code, e.Message)
}

func (c *Client) do(ctx context.Context, method, path string, body any) (json.RawMessage, error) {
	var rd io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		rd = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.Base+path, rd)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	hc := c.HTTP
	if hc == nil {
		hc = http.DefaultClient
	}
	resp, err := hc.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxResponse+1))
	if err != nil {
		return nil, err
	}
	if len(data) > maxResponse {
		return nil, errors.New("webdriver: response too large")
	}
	var env struct {
		Value json.RawMessage `json:"value"`
	}
	if err := json.Unmarshal(data, &env); err != nil {
		return nil, fmt.Errorf("webdriver: %d, not JSON: %w", resp.StatusCode, err)
	}
	if resp.StatusCode >= 400 {
		var e struct{ Error, Message string }
		json.Unmarshal(env.Value, &e)
		return nil, &Error{Status: resp.StatusCode, Code: e.Error, Message: e.Message}
	}
	return env.Value, nil
}

// Session is one browser session.
type Session struct {
	c  *Client
	ID string
	// Version is the browser's version the driver reported (capabilities
	// browserVersion), "" if it did not say.
	Version string
}

// NewSession creates a session; caps is the alwaysMatch capabilities object.
func (c *Client) NewSession(ctx context.Context, caps map[string]any) (*Session, error) {
	v, err := c.do(ctx, http.MethodPost, "/session", map[string]any{"capabilities": map[string]any{"alwaysMatch": caps}})
	if err != nil {
		return nil, err
	}
	var r struct {
		SessionID    string `json:"sessionId"`
		Capabilities struct {
			BrowserVersion string `json:"browserVersion"`
		} `json:"capabilities"`
	}
	if err := json.Unmarshal(v, &r); err != nil || r.SessionID == "" {
		return nil, fmt.Errorf("webdriver: no session id in %.200s", v)
	}
	return &Session{c: c, ID: r.SessionID, Version: r.Capabilities.BrowserVersion}, nil
}

// Navigate loads url.
func (s *Session) Navigate(ctx context.Context, url string) error {
	_, err := s.c.do(ctx, http.MethodPost, "/session/"+s.ID+"/url", map[string]string{"url": url})
	return err
}

// Execute runs a script in the page and returns its JSON result.
func (s *Session) Execute(ctx context.Context, script string) (json.RawMessage, error) {
	return s.c.do(ctx, http.MethodPost, "/session/"+s.ID+"/execute/sync", map[string]any{"script": script, "args": []any{}})
}

// Delete ends the session.
func (s *Session) Delete(ctx context.Context) error {
	_, err := s.c.do(ctx, http.MethodDelete, "/session/"+s.ID, nil)
	return err
}

// Start runs driver (for example /usr/bin/safaridriver) on a free port and
// waits for GET /status to answer. stop kills the driver by PID.
func Start(ctx context.Context, driver string) (c *Client, stop func(), err error) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, nil, err
	}
	port := ln.Addr().(*net.TCPAddr).Port
	ln.Close()
	cmd := exec.Command(driver, "-p", fmt.Sprint(port))
	if err := cmd.Start(); err != nil {
		return nil, nil, err
	}
	stop = func() { cmd.Process.Kill(); cmd.Wait() }
	c = &Client{Base: fmt.Sprintf("http://127.0.0.1:%d", port), HTTP: &http.Client{Timeout: 2 * time.Minute}}
	deadline := time.Now().Add(15 * time.Second)
	for {
		if _, err := c.do(ctx, http.MethodGet, "/status", nil); err == nil {
			return c, stop, nil
		}
		if time.Now().After(deadline) || ctx.Err() != nil {
			stop()
			return nil, nil, fmt.Errorf("webdriver: %s did not answer /status on port %d", driver, port)
		}
		time.Sleep(100 * time.Millisecond)
	}
}
