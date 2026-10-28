package webgpu

// WGSL compute kernels for the ONNX operators, one invocation per output
// element (per row for Softmax). Shapes and strides travel in a "meta"
// storage buffer of u32 (floats as their bits), so one compiled pipeline
// serves every shape. Dispatches lay invocations over x and y; the linear
// index is x + y * 65535 * 64 (jsgpu.Dispatch).
//
// Every kernel is checked against the CPU operators of package onnx on the
// ONNX node tests (e2e, in a browser).

// MaxRank is the largest tensor rank the kernels index.
const MaxRank = 8

// MaxSpatial is the largest number of spatial axes of Conv and pooling.
const MaxSpatial = 3

const prelude = `
@group(0) @binding(0) var<storage, read> prm: array<u32>;
fn gid(g: vec3<u32>) -> u32 { return g.x + g.y * 4194240u; }
// -inf, computed at run time: a constant expression that evaluates to an
// infinity is a shader-creation error in WGSL.
fn ninf() -> f32 { return bitcast<f32>(0xff800000u | (prm[0] & 0u)); }
`

// binaryKernel: meta = n, rank, out[8], aStride[8], bStride[8].
func binaryKernel(op string) string {
	return prelude + `
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = gid(g);
  if (i >= prm[0]) { return; }
  var rem = i; var ia = 0u; var ib = 0u;
  for (var d = i32(prm[1]) - 1; d >= 0; d--) {
    let dim = prm[2u + u32(d)];
    let c = rem % dim; rem = rem / dim;
    ia += c * prm[10u + u32(d)]; ib += c * prm[18u + u32(d)];
  }
  let p = a[ia]; let q = b[ib];
  y[i] = ` + op + `;
}`
}

var binaryOps = map[string]string{"Add": "p + q", "Sub": "p - q", "Mul": "p * q", "Div": "p / q"}

// reluKernel: meta = n.
const reluKernel = prelude + `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = gid(g);
  if (i >= prm[0]) { return; }
  y[i] = max(x[i], 0.0);
}`

// matmulKernel covers MatMul and Gemm: meta = n, M, N, K, as0, as1, bs0,
// bs1, cs0, cs1, hasC, alpha, beta, batchRank, outBatch[8],
// aBatch[8], bBatch[8] (element strides of each operand's batch).
const matmulKernel = prelude + `
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> c: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = gid(g);
  if (i >= prm[0]) { return; }
  let M = prm[1]; let N = prm[2]; let K = prm[3];
  let n = i % N; let m = (i / N) % M;
  var rem = i / (M * N); var ab = 0u; var bb = 0u;
  for (var d = i32(prm[13]) - 1; d >= 0; d--) {
    let dim = prm[14u + u32(d)];
    let co = rem % dim; rem = rem / dim;
    ab += co * prm[22u + u32(d)]; bb += co * prm[30u + u32(d)];
  }
  var acc = 0.0;
  for (var k = 0u; k < K; k++) {
    acc += a[ab + m * prm[4] + k * prm[5]] * b[bb + k * prm[6] + n * prm[7]];
  }
  var v = bitcast<f32>(prm[11]) * acc;
  if (prm[10] != 0u) { v += bitcast<f32>(prm[12]) * c[m * prm[8] + n * prm[9]]; }
  y[i] = v;
}`

// windowMeta layout shared by conv and pooling, after the kernel's own
// head: in[3], kernel[3], stride[3], dil[3], padB[3], padE[3], out[3]
// (unused spatial axes as 1 / 0).

// convKernel: meta = n, N, C, M, Cg, Mg, hasBias, then the window at 7.
const convKernel = prelude + `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
fn wm(f: u32, d: u32) -> u32 { return prm[7u + f * 3u + d]; }
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = gid(g);
  if (i >= prm[0]) { return; }
  let C = prm[2]; let M = prm[3]; let Cg = prm[4]; let Mg = prm[5];
  let o2 = i % wm(6u, 2u); let o1 = (i / wm(6u, 2u)) % wm(6u, 1u); let o0 = (i / (wm(6u, 2u) * wm(6u, 1u))) % wm(6u, 0u);
  let outPlane = wm(6u, 0u) * wm(6u, 1u) * wm(6u, 2u);
  let m = (i / outPlane) % M; let ni = i / (outPlane * M);
  let grp = m / Mg;
  let inPlane = wm(0u, 0u) * wm(0u, 1u) * wm(0u, 2u);
  let kPlane = wm(1u, 0u) * wm(1u, 1u) * wm(1u, 2u);
  var acc = 0.0;
  if (prm[6] != 0u) { acc = bias[m]; }
  for (var c = 0u; c < Cg; c++) {
    let xb = (ni * C + grp * Cg + c) * inPlane;
    let wb = (m * Cg + c) * kPlane;
    for (var k0 = 0u; k0 < wm(1u, 0u); k0++) {
      let p0 = i32(o0 * wm(2u, 0u) + k0 * wm(3u, 0u)) - i32(wm(4u, 0u));
      if (p0 < 0 || p0 >= i32(wm(0u, 0u))) { continue; }
      for (var k1 = 0u; k1 < wm(1u, 1u); k1++) {
        let p1 = i32(o1 * wm(2u, 1u) + k1 * wm(3u, 1u)) - i32(wm(4u, 1u));
        if (p1 < 0 || p1 >= i32(wm(0u, 1u))) { continue; }
        for (var k2 = 0u; k2 < wm(1u, 2u); k2++) {
          let p2 = i32(o2 * wm(2u, 2u) + k2 * wm(3u, 2u)) - i32(wm(4u, 2u));
          if (p2 < 0 || p2 >= i32(wm(0u, 2u))) { continue; }
          let xi = xb + (u32(p0) * wm(0u, 1u) + u32(p1)) * wm(0u, 2u) + u32(p2);
          let wi = wb + (k0 * wm(1u, 1u) + k1) * wm(1u, 2u) + k2;
          acc += x[xi] * w[wi];
        }
      }
    }
  }
  y[i] = acc;
}`

// poolKernel: meta = n, planes (N*C), isMax, includePad, then the window
// at 4.
const poolKernel = prelude + `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
fn wm(f: u32, d: u32) -> u32 { return prm[4u + f * 3u + d]; }
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = gid(g);
  if (i >= prm[0]) { return; }
  let isMax = prm[2] != 0u; let incl = prm[3] != 0u;
  let o2 = i % wm(6u, 2u); let o1 = (i / wm(6u, 2u)) % wm(6u, 1u); let o0 = (i / (wm(6u, 2u) * wm(6u, 1u))) % wm(6u, 0u);
  let outPlane = wm(6u, 0u) * wm(6u, 1u) * wm(6u, 2u);
  let plane = i / outPlane;
  let xb = plane * wm(0u, 0u) * wm(0u, 1u) * wm(0u, 2u);
  var best = ninf();
  var sum = 0.0; var cnt = 0.0; var padCnt = 0.0;
  for (var k0 = 0u; k0 < wm(1u, 0u); k0++) {
    let p0 = i32(o0 * wm(2u, 0u) + k0 * wm(3u, 0u)) - i32(wm(4u, 0u));
    let in0 = p0 >= 0 && p0 < i32(wm(0u, 0u));
    let pad0 = p0 < i32(wm(0u, 0u) + wm(5u, 0u));
    for (var k1 = 0u; k1 < wm(1u, 1u); k1++) {
      let p1 = i32(o1 * wm(2u, 1u) + k1 * wm(3u, 1u)) - i32(wm(4u, 1u));
      let in1 = p1 >= 0 && p1 < i32(wm(0u, 1u));
      let pad1 = p1 < i32(wm(0u, 1u) + wm(5u, 1u));
      for (var k2 = 0u; k2 < wm(1u, 2u); k2++) {
        let p2 = i32(o2 * wm(2u, 2u) + k2 * wm(3u, 2u)) - i32(wm(4u, 2u));
        let in2 = p2 >= 0 && p2 < i32(wm(0u, 2u));
        let pad2 = p2 < i32(wm(0u, 2u) + wm(5u, 2u));
        if (pad0 && pad1 && pad2) { padCnt += 1.0; }
        if (!(in0 && in1 && in2)) { continue; }
        let v = x[xb + (u32(p0) * wm(0u, 1u) + u32(p1)) * wm(0u, 2u) + u32(p2)];
        if (v > best || v != v) { best = v; }
        sum += v; cnt += 1.0;
      }
    }
  }
  if (isMax) { y[i] = best; return; }
  var d = cnt;
  if (incl) { d = padCnt; }
  if (d > 0.0) { y[i] = sum / d; } else { y[i] = 0.0; }
}`

// softmaxKernel: one invocation per row; meta = rows, len, inner.
const softmaxKernel = prelude + `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let r = gid(g);
  if (r >= prm[0]) { return; }
  let L = prm[1]; let inner = prm[2];
  let base = (r / inner) * L * inner + r % inner;
  var mx = ninf();
  for (var j = 0u; j < L; j++) { mx = max(mx, x[base + j * inner]); }
  var s = 0.0;
  for (var j = 0u; j < L; j++) { let e = exp(x[base + j * inner] - mx); y[base + j * inner] = e; s += e; }
  for (var j = 0u; j < L; j++) { y[base + j * inner] = y[base + j * inner] / s; }
}`

// copyKernel: strided copy over an index space (Transpose, a Concat
// part): meta = n, rank, shape[8], srcStride[8], dstStride[8], srcOff,
// dstOff.
const copyKernel = prelude + `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = gid(g);
  if (i >= prm[0]) { return; }
  var rem = i; var s = prm[26]; var t = prm[27];
  for (var d = i32(prm[1]) - 1; d >= 0; d--) {
    let dim = prm[2u + u32(d)];
    let c = rem % dim; rem = rem / dim;
    s += c * prm[10u + u32(d)]; t += c * prm[18u + u32(d)];
  }
  y[t] = x[s];
}`
