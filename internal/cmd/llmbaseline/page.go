package main

import "encoding/json"

func decodeMap(m map[string]any, v any) error {
	b, err := json.Marshal(m)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, v)
}

// pageHTML runs one WebLLM case. The bundle, the model libraries and the
// weights all come from the local server (pinned files), never the network.
const pageHTML = `<!doctype html>
<meta charset="utf-8"><title>llm baseline</title>
<script type="module">
import * as webllm from "/webllm.js";
window.runCase = async (a) => {
  const r = {};
  try {
    const ad = await navigator.gpu.requestAdapter();
    if (!ad) { r.error = "no adapter"; return r; }
    const info = ad.info || {};
    r.adapter = {vendor: info.vendor || "", architecture: info.architecture || "",
      description: info.description || "", fallback: !!(info.isFallbackAdapter || ad.isFallbackAdapter)};
    r.f16 = ad.features.has("shader-f16");
    const appConfig = {model_list: [{
      model: location.origin + "/mlc/" + a.id, model_id: a.id,
      model_lib: location.origin + "/lib/" + a.lib,
      required_features: a.f16 ? ["shader-f16"] : [],
      overrides: {context_window_size: 4096}}]};
    const engine = await webllm.CreateMLCEngine(a.id, {appConfig});
    // Warm-up: pipeline creation and first-use costs stay out of the measure.
    await engine.completions.create({prompt: "the the the the", max_tokens: 8, ignore_eos: true, temperature: 0});
    await engine.resetChat();
    const res = await engine.completions.create({prompt: a.prompt, max_tokens: a.tg, ignore_eos: true, temperature: 0});
    const u = res.usage;
    r.promptTokens = u.prompt_tokens; r.completionTokens = u.completion_tokens;
    r.prefillTps = u.extra.prefill_tokens_per_s; r.decodeTps = u.extra.decode_tokens_per_s;
    await engine.unload();
  } catch (e) { r.error = String(e && e.stack || e); }
  return r;
};
window.ready = true;
</script>`
