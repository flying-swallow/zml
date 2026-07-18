# Caliper

[![CI](https://github.com/flying-swallow/zig-linear-algebra/actions/workflows/ci.yml/badge.svg)](https://github.com/flying-swallow/zig-linear-algebra/actions/workflows/ci.yml)

A high-performance SIMD linear algebra, geometry, and GJK/EPA collision library for Zig, providing vector, matrix, quaternion, and geometry operations.

## Features

- **Vectors** (`caliper.vec`) — norm, dot, cross, normalize, reflect, distance, angle, swizzle, fused `sin_cos`, and more.
- **Matrices** (`caliper.Mat`) — generic column-major `Mat(T, cols, rows)` (`items: [cols][rows]T`, GPU-upload ready) with multiply, transforms, etc.
- **Quaternions** (`caliper.quat`) — rotation, slerp/nlerp, axis-angle and Euler conversions.
- **Geometry** (`caliper.geom`) — AABB, sphere, plane, capsule, OBB, ray, frustum, overlap/containment tests.
- Extras: `caliper.scalar` (clamp/lerp/smoothstep), `caliper.color`, `caliper.packing`, `caliper.random`.
- Built on Zig's native `@Vector` SIMD types — **but functions also accept plain arrays** (see below).

## Installation

```zig
zig fetch --save "git+https://github.com/flying-swallow/zig-linear-algebra.git"
```

Then in your `build.zig`:

```zig
const caliper = b.dependency("caliper", .{
    .target = target,
    .optimize = optimize,
});

exe.root_module.addImport("caliper", caliper.module("caliper"));
```

## Usage

```zig
const caliper = @import("caliper");

const a = caliper.Vec3f32{ 1, 2, 3 }; // @Vector(3, f32)
const b = caliper.Vec3f32{ 4, 5, 6 };

const d = caliper.vec.dot(a, b);       // f32 = 32
const c = caliper.vec.cross(a, b);     // @Vector(3, f32)
const n = caliper.vec.normalize(a);    // @Vector(3, f32)
const r = caliper.vec.sin_cos(a);      // .{ .sin_out, .cos_out }
```

## `@Vector` and array inputs

Every length-generic vector function accepts **both** a native `@Vector(N, T)` and a plain
`[N]T` array, and the return **preserves the input's container kind** (array in → array out,
vector in → vector out):

```zig
const arr = [3]f32{ 1, 2, 3 };
const d  = caliper.vec.dot(arr, arr);     // f32
const nz = caliper.vec.normalize(arr);    // [3]f32
```

Arrays are meant for arbitrarily long data. Rather than coercing a large array into one wide
`@Vector(N, T)` — which makes LLVM emit a single enormous SIMD instruction and bloats the binary
— array inputs are processed in **chunks sized to the CPU's native SIMD width**
(`std.simd.suggestVectorLengthForCpu`) via a compact loop. `@Vector` inputs keep the single-op
path (you chose that width explicitly).

For example, `norm` over a `[245]f32` compiles to **324 bytes** via the chunked path, versus
**2,529 bytes** for the equivalent `@Vector(245, f32)` op (`-OReleaseSmall`) — and it is faster
too (see below).

## Benchmarks

Run the benchmark suite (uses [zBench](https://github.com/hendriknielaender/zBench)):

```sh
zig build bench -Doptimize=ReleaseFast
```

Representative results (`ReleaseFast`; absolute numbers are machine-dependent — the point is the
array-vs-`@Vector` ratio at large `N`):

| Operation (N = 245)        | array (chunked) | `@Vector(N)` (single op) | speedup |
| -------------------------- | --------------: | -----------------------: | ------: |
| `norm`                     |          37 ns  |                  235 ns  |   6.3×  |
| `dot`                      |          40 ns  |                  313 ns  |   7.8×  |
| `normalize`                |          57 ns  |                  261 ns  |   4.6×  |

| Sin/Cos over 256 elements  |   time/run |
| -------------------------- | ---------: |
| scalar `std.math` loop     |    695 ns  |
| `sin_cos` (`@Vector`)      |    106 ns  |
| `sin_cos` (array, chunked) |    139 ns  |

For the length-generic reductions/maps, the chunked array path is both **smaller and faster**
than one wide `@Vector` op at large `N`: the wide op forces LLVM into a slow, bloated instruction
sequence, while the native-width loop stays compact and vectorized. `sin_cos` is pure
element-wise, so the `@Vector` form already vectorizes cleanly and the two are comparable.

## Testing

```sh
zig build test
```
