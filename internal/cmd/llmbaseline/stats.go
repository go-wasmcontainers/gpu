package main

import (
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// Sample is one measured round of one configuration.
type Sample struct {
	Config  string  `json:"config"`
	Round   int     `json:"round"`
	Load1   float64 `json:"load1"` // 1-minute load just before the run
	PP      float64 `json:"pp_tps"`
	TG      float64 `json:"tg_tps"`
	PPToks  int     `json:"pp_tokens"`
	TGToks  int     `json:"tg_tokens"`
	Adapter string  `json:"adapter,omitempty"`
	Note    string  `json:"note,omitempty"`
}

var loadRE = regexp.MustCompile(`load averages?:\s*([0-9]+[.,][0-9]+)`)

// ParseLoad returns the 1-minute load average from uptime's output.
func ParseLoad(s string) (float64, error) {
	m := loadRE.FindStringSubmatch(s)
	if m == nil {
		return 0, fmt.Errorf("no load average in %q", strings.TrimSpace(s))
	}
	return strconv.ParseFloat(strings.Replace(m[1], ",", ".", 1), 64)
}

// Percentile is the linear-interpolation percentile (NumPy's default).
func Percentile(xs []float64, p float64) float64 {
	if len(xs) == 0 {
		return math.NaN()
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	pos := p / 100 * float64(len(s)-1)
	lo := int(math.Floor(pos))
	hi := int(math.Ceil(pos))
	return s[lo] + (s[hi]-s[lo])*(pos-float64(lo))
}

type benchRow struct {
	NPrompt int     `json:"n_prompt"`
	NGen    int     `json:"n_gen"`
	AvgTS   float64 `json:"avg_ts"`
}

// ParseBench reads `llama-bench -o json`: one row for pp, one for tg.
func ParseBench(b []byte) (pp float64, ppN int, tg float64, tgN int, err error) {
	var rows []benchRow
	if err = json.Unmarshal(b, &rows); err != nil {
		return
	}
	for _, r := range rows {
		switch {
		case r.NPrompt > 0 && r.NGen == 0:
			pp, ppN = r.AvgTS, r.NPrompt
		case r.NGen > 0 && r.NPrompt == 0:
			tg, tgN = r.AvgTS, r.NGen
		}
	}
	if pp == 0 || tg == 0 {
		err = fmt.Errorf("llama-bench output has no pp and tg rows: %s", b)
	}
	return
}

// Table renders p50 and the min-max spread per configuration, in order.
func Table(samples []Sample, order []string) string {
	by := map[string][]Sample{}
	for _, s := range samples {
		by[s.Config] = append(by[s.Config], s)
	}
	var b strings.Builder
	b.WriteString("| Configuration | n | prefill p50 | prefill min-max | decode p50 | decode min-max | max load |\n")
	b.WriteString("|---|---|---|---|---|---|---|\n")
	for _, name := range order {
		ss := by[name]
		if len(ss) == 0 {
			continue
		}
		var pp, tg []float64
		maxLoad := 0.0
		for _, s := range ss {
			pp = append(pp, s.PP)
			tg = append(tg, s.TG)
			maxLoad = math.Max(maxLoad, s.Load1)
		}
		fmt.Fprintf(&b, "| %s | %d | %.1f | %.1f - %.1f | %.1f | %.1f - %.1f | %.2f |\n", name, len(ss),
			Percentile(pp, 50), Percentile(pp, 0), Percentile(pp, 100),
			Percentile(tg, 50), Percentile(tg, 0), Percentile(tg, 100), maxLoad)
	}
	return b.String()
}
