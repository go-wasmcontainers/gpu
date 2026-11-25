//go:build !wasip1

package wasigpu

import (
	"errors"
	"sync/atomic"
	"testing"
)

// The calls of a frame are replaced by fakes that answer a handle, but for one
// that is missing and fails with ErrUnavailable: the frame must report it
// wherever it comes and release what it had made.

type step struct{ module, name string }

// recordSteps are the calls of the render pass, in order.
var recordSteps = []step{
	{contextModule, "[method]context.get-current-texture"},
	{webgpuModule, "[method]gpu-texture.create-view"},
	{webgpuModule, "[method]gpu-device.create-command-encoder"},
	{webgpuModule, "[method]gpu-command-encoder.begin-render-pass"},
	{webgpuModule, "[method]gpu-render-pass-encoder.set-pipeline"},
	{webgpuModule, "[method]gpu-render-pass-encoder.draw"},
	{webgpuModule, "[method]gpu-render-pass-encoder.end"},
	{webgpuModule, "[method]gpu-command-encoder.finish"},
}

var (
	submitStep  = step{webgpuModule, "[method]gpu-queue.submit"}
	presentStep = step{contextModule, "[method]context.present"}
	beginStep   = step{contextModule, "[method]context.submit-present-begin"}
	endStep     = step{contextModule, "[method]context.submit-present-end"}
)

func frameSteps() []step {
	all := append([]step(nil), recordSteps...)
	return append(all, submitStep, presentStep, beginStep, endStep)
}

// install makes every step of all answer a handle but the one in missing, and
// counts the resource drops.
func install(t *testing.T, missing *step, all []step) *int {
	t.Helper()
	savedIn, savedCalls := callIn, calls
	t.Cleanup(func() { callIn, calls = savedIn, savedCalls })
	have := map[step]bool{}
	for _, s := range all {
		have[s] = true
	}
	if missing != nil {
		have[*missing] = false
	}
	callIn = func(module, name string, args ...any) (any, error) {
		if !have[step{module, name}] {
			return nil, ErrUnavailable
		}
		return uint32(0), nil
	}
	drops := new(int)
	calls = map[string]func([]uint64) uint64{}
	for _, r := range []string{"gpu-texture", "gpu-texture-view", "gpu-command-encoder", "gpu-render-pass-encoder", "gpu-command-buffer"} {
		calls[webgpuModule+" [resource-drop]"+r] = func([]uint64) uint64 { *drops++; return 0 }
	}
	return drops
}

func testSurface() (*Surface, *RenderPipeline) {
	return &Surface{d: &Device{dev: 1, queue: 2, format: 3}, ctx: 4}, &RenderPipeline{pipe: 5, layout: 6}
}

func TestFrameAndBeginFrameReportEveryStepThatFails(t *testing.T) {
	all := frameSteps()
	for _, missing := range all {
		t.Run(missing.name, func(t *testing.T) {
			install(t, &missing, all)
			s, p := testSurface()
			frameFails := missing != beginStep && missing != endStep
			if err := s.Frame(p, nil, [4]float64{}, 3); (err != nil) != frameFails {
				t.Errorf("Frame with %s missing: %v", missing.name, err)
			}
			f, err := s.BeginFrame(p, nil, [4]float64{}, 3)
			beginFails := missing == beginStep
			for _, r := range recordSteps {
				beginFails = beginFails || missing == r
			}
			if (err != nil) != beginFails {
				t.Fatalf("BeginFrame with %s missing: %v", missing.name, err)
			}
			if err != nil {
				if !errors.Is(err, ErrUnavailable) || f != nil {
					t.Errorf("BeginFrame: %v, %v; want ErrUnavailable and no frame", f, err)
				}
				return
			}
			if endErr := f.End(); (endErr != nil) != (missing == endStep) {
				t.Errorf("End with %s missing: %v", missing.name, endErr)
			}
		})
	}
}

func TestFrameReleasesWhatItMadeWhenAStepFails(t *testing.T) {
	missing := submitStep
	drops := install(t, &missing, frameSteps())
	s, p := testSurface()
	if err := s.Frame(p, nil, [4]float64{}, 3); err == nil {
		t.Fatal("Frame with no submit succeeded")
	}
	// texture, view, encoder, pass, command buffer
	if *drops != 5 {
		t.Errorf("Frame released %d objects after a failed submit, want 5", *drops)
	}
}

func TestPendingFrameReadyIsTheWordTheHostSets(t *testing.T) {
	drops := install(t, nil, frameSteps())
	s, p := testSurface()
	f, err := s.BeginFrame(p, nil, [4]float64{}, 3)
	if err != nil {
		t.Fatal(err)
	}
	if f.Ready() {
		t.Fatal("Ready before the host stored 1")
	}
	atomic.StoreUint32(f.done, 1)
	if !f.Ready() {
		t.Error("not Ready after the host stored 1")
	}
	if *drops != 0 {
		t.Errorf("%d objects released before End", *drops)
	}
	if err := f.End(); err != nil {
		t.Fatal(err)
	}
	if *drops != 5 {
		t.Errorf("End released %d objects, want 5", *drops)
	}
}
