package main

import (
	"context"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"hash"
	"io"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"strings"
)

// hfBase is replaced by the tests.
var hfBase = "https://huggingface.co"

// Pinned is one downloaded, verified file: what the results doc quotes.
type Pinned struct {
	Repo     string `json:"repo"`
	Commit   string `json:"commit"`
	Path     string `json:"path"`
	Size     int64  `json:"size"`
	SHA256   string `json:"sha256"`
	Verified string `json:"verified"` // "lfs-sha256" or "git-blob-sha1": what the file was checked against
}

type hfEntry struct {
	Type string `json:"type"`
	OID  string `json:"oid"`
	Size int64  `json:"size"`
	Path string `json:"path"`
	LFS  *struct {
		OID  string `json:"oid"`
		Size int64  `json:"size"`
	} `json:"lfs"`
}

func getJSON(ctx context.Context, u string, v any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GET %s: %s", u, resp.Status)
	}
	return json.NewDecoder(io.LimitReader(resp.Body, 64<<20)).Decode(v)
}

// validRepo accepts "owner/name" only.
func validRepo(r string) bool {
	o, n, ok := strings.Cut(r, "/")
	return ok && o != "" && n != "" && !strings.Contains(n, "/") && filepath.IsLocal(o) && filepath.IsLocal(n)
}

// FetchHF resolves repo's current commit, then downloads each wanted file
// (all files when want is empty) from that commit into dir/<repo>/<path>,
// checking size and hash against what the Hub lists. A file already present
// and correct is not downloaded again.
func FetchHF(ctx context.Context, dir, repo string, want []string) ([]Pinned, error) {
	if !validRepo(repo) {
		return nil, fmt.Errorf("bad repo %q", repo)
	}
	var info struct {
		SHA string `json:"sha"`
	}
	if err := getJSON(ctx, hfBase+"/api/models/"+repo, &info); err != nil {
		return nil, err
	}
	if len(info.SHA) != 40 {
		return nil, fmt.Errorf("%s: no commit sha in the answer", repo)
	}
	var tree []hfEntry
	if err := getJSON(ctx, hfBase+"/api/models/"+repo+"/tree/"+info.SHA+"?recursive=true&limit=1000", &tree); err != nil {
		return nil, err
	}
	wanted := map[string]bool{}
	for _, w := range want {
		wanted[w] = true
	}
	var out []Pinned
	for _, e := range tree {
		if e.Type != "file" || (len(want) > 0 && !wanted[e.Path]) {
			continue
		}
		delete(wanted, e.Path)
		p, err := fetchOne(ctx, dir, repo, info.SHA, e)
		if err != nil {
			return out, fmt.Errorf("%s/%s: %w", repo, e.Path, err)
		}
		out = append(out, p)
	}
	for w := range wanted {
		return out, fmt.Errorf("%s: no file %q at %s", repo, w, info.SHA)
	}
	return out, nil
}

func hashFile(name string, size int64, gitBlob bool) (sha256hex, sha1hex string, err error) {
	f, err := os.Open(name)
	if err != nil {
		return "", "", err
	}
	defer f.Close()
	return hashReader(f, size, gitBlob)
}

func hashReader(r io.Reader, size int64, gitBlob bool) (string, string, error) {
	h2 := sha256.New()
	var h1 hash.Hash = sha1.New()
	if gitBlob {
		fmt.Fprintf(h1, "blob %d\x00", size)
	}
	if _, err := io.Copy(io.MultiWriter(h2, h1), r); err != nil {
		return "", "", err
	}
	return hex.EncodeToString(h2.Sum(nil)), hex.EncodeToString(h1.Sum(nil)), nil
}

func check(e hfEntry, size int64, s256, s1 string) (string, error) {
	if e.LFS != nil {
		if size != e.LFS.Size {
			return "", fmt.Errorf("size %d, the Hub lists %d", size, e.LFS.Size)
		}
		if s256 != e.LFS.OID {
			return "", fmt.Errorf("sha256 %s, the Hub lists %s", s256, e.LFS.OID)
		}
		return "lfs-sha256", nil
	}
	if size != e.Size {
		return "", fmt.Errorf("size %d, the Hub lists %d", size, e.Size)
	}
	if s1 != e.OID {
		return "", fmt.Errorf("git blob sha1 %s, the Hub lists %s", s1, e.OID)
	}
	return "git-blob-sha1", nil
}

func fetchOne(ctx context.Context, dir, repo, commit string, e hfEntry) (Pinned, error) {
	if !filepath.IsLocal(e.Path) {
		return Pinned{}, fmt.Errorf("unsafe path %q", e.Path)
	}
	dst := filepath.Join(dir, filepath.FromSlash(repo), filepath.FromSlash(e.Path))
	want := e.Size
	if e.LFS != nil {
		want = e.LFS.Size
	}
	if st, err := os.Stat(dst); err == nil && st.Size() == want {
		s256, s1, err := hashFile(dst, want, e.LFS == nil)
		if err == nil {
			if v, err := check(e, st.Size(), s256, s1); err == nil {
				return Pinned{repo, commit, e.Path, st.Size(), s256, v}, nil
			}
		}
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return Pinned{}, err
	}
	u := hfBase + "/" + repo + "/resolve/" + commit + "/" + (&url.URL{Path: e.Path}).EscapedPath()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return Pinned{}, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return Pinned{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return Pinned{}, fmt.Errorf("GET %s: %s", u, resp.Status)
	}
	part := dst + ".part"
	f, err := os.Create(part)
	if err != nil {
		return Pinned{}, err
	}
	// Bounded by the size the Hub listed, plus one byte to detect a longer body.
	n, err := io.Copy(f, io.LimitReader(resp.Body, want+1))
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return Pinned{}, err
	}
	s256, s1, err := hashFile(part, n, e.LFS == nil)
	if err != nil {
		return Pinned{}, err
	}
	v, err := check(e, n, s256, s1)
	if err != nil {
		_ = os.Remove(part)
		return Pinned{}, err
	}
	if err := os.Rename(part, dst); err != nil {
		return Pinned{}, err
	}
	return Pinned{repo, commit, e.Path, n, s256, v}, nil
}

// FetchURL downloads u into dst and returns its size and sha256. When want is
// not empty the hash must equal it. Used for the WebLLM model libraries,
// fetched from a GitHub commit URL.
func FetchURL(ctx context.Context, u, dst, want string) (int64, string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return 0, "", err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return 0, "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, "", fmt.Errorf("GET %s: %s", u, resp.Status)
	}
	if err := os.MkdirAll(path.Dir(filepath.ToSlash(dst)), 0o755); err != nil {
		return 0, "", err
	}
	part := dst + ".part"
	f, err := os.Create(part)
	if err != nil {
		return 0, "", err
	}
	n, err := io.Copy(f, io.LimitReader(resp.Body, 1<<30))
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return 0, "", err
	}
	s256, _, err := hashFile(part, n, false)
	if err != nil {
		return 0, "", err
	}
	if want != "" && s256 != want {
		_ = os.Remove(part)
		return n, s256, fmt.Errorf("sha256 %s, want %s", s256, want)
	}
	return n, s256, os.Rename(part, dst)
}
