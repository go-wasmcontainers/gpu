package main

import (
	"bytes"
	"testing"
)

func TestSweep(t *testing.T) {
	rs, err := Sweep(3, 6, func(k int, w uint64) (bool, error) {
		if w != 1<<k {
			t.Errorf("work %d for k=%d", w, k)
		}
		return k >= 5, nil
	})
	if err != nil || len(rs) != 4 || Smallest(rs) != 5 {
		t.Fatalf("%v %v %d", rs, err, Smallest(rs))
	}
	if Smallest([]Result{{K: 1}}) != -1 {
		t.Error("no pass must be -1")
	}
	if _, err := Sweep(5, 3, nil); err == nil {
		t.Error("empty range accepted")
	}
}

func TestRootCmdRequiresFlags(t *testing.T) {
	c := RootCmd()
	var out bytes.Buffer
	c.SetOut(&out)
	c.SetErr(&out)
	c.SetArgs(nil)
	if err := c.Execute(); err == nil {
		t.Error("ran without --run and --out")
	}
}
