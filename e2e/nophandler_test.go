//go:build e2e

package e2e

import "net/http"

// nopHandler serves nothing: the capability probe page is built into serve.
type nopHandler struct{}

func (nopHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) { http.NotFound(w, r) }
