//! Chunked SIMD kernels for ARRAY inputs.
//!
//! `vector.zig` accepts both `@Vector(N,T)` and `[N]T`. A `@Vector` input is
//! processed as a single native op (the caller chose that width). An array,
//! however, may be arbitrarily long (`[245]f32`); coercing it to one wide
//! `@Vector(245,f32)` makes LLVM emit one enormous SIMD op and bloats the
//! binary. These kernels instead walk an array in chunks sized by the CPU's
//! native SIMD width (`laneCount`), using a runtime loop over full chunks plus
//! one comptime-sized remainder chunk, so no wide vector is ever materialized.

const std = @import("std");
const builtin = @import("builtin");
const meta = @import("meta.zig");

/// Native SIMD width (lane count) for element type `T` on the target CPU, or 1
/// when the target has no vector unit for `T`. This is the chunk size used by
/// every kernel below.
pub fn laneCount(comptime T: type) comptime_int {
    return std.simd.suggestVectorLengthForCpu(T, builtin.cpu) orelse 1;
}

/// Sum of `a[i] * b[i]`, accumulated in compute type `C`.
///
/// `C` must be either `Child(a)` itself (identity coercion) or a WIDER float
/// (element-wise float widen) — never an int->float change, which is illegal on
/// a vector. The int path keeps `C == T` and the caller applies `@floatFromInt`
/// once to the scalar result, matching `vector.norm_sqr_adv`.
pub fn sumProducts(comptime C: type, a: anytype, b: anytype) C {
    const T = meta.Child(@TypeOf(a));
    const N = comptime meta.lengthOf(@TypeOf(a));
    const W = comptime laneCount(T);
    const full = comptime (N / W) * W; // largest multiple of W <= N
    const rem = comptime N % W; // leftover lanes

    // Bind to const locals so `x[i..]` (runtime i) is addressable.
    const ba = a;
    const bb = b;

    var acc: @Vector(W, C) = @splat(0);
    var i: usize = 0;
    while (i < full) : (i += W) { // runtime loop -> bounded code size
        const ca: @Vector(W, T) = ba[i..][0..W].*;
        const cb: @Vector(W, T) = bb[i..][0..W].*;
        const va: @Vector(W, C) = ca; // identity or float-widen
        const vb: @Vector(W, C) = cb;
        acc += va * vb;
    }
    var total: C = @reduce(.Add, acc);
    if (rem != 0) { // comptime guard: @Vector(0, C) is illegal
        const ca: @Vector(rem, T) = ba[full..][0..rem].*;
        const cb: @Vector(rem, T) = bb[full..][0..rem].*;
        const va: @Vector(rem, C) = ca;
        const vb: @Vector(rem, C) = cb;
        total += @reduce(.Add, va * vb);
    }
    return total;
}

/// Sum of `a[i] * a[i]`, accumulated in compute type `C`.
pub fn sumSquares(comptime C: type, a: anytype) C {
    return sumProducts(C, a, a);
}

/// Apply width-generic `op(chunk, ctx)` to every native-width chunk of `a`,
/// returning a fresh array of the same element type and length. `op` must be
/// generic over the chunk width: `fn (chunk: anytype, ctx) @TypeOf(chunk)`.
pub fn mapUnary(a: anytype, ctx: anytype, comptime op: anytype) [meta.lengthOf(@TypeOf(a))]meta.Child(@TypeOf(a)) {
    const T = meta.Child(@TypeOf(a));
    const N = comptime meta.lengthOf(@TypeOf(a));
    const W = comptime laneCount(T);
    const full = comptime (N / W) * W;
    const rem = comptime N % W;

    const ba = a;
    var out: [N]T = undefined;
    var i: usize = 0;
    while (i < full) : (i += W) {
        const c: @Vector(W, T) = ba[i..][0..W].*;
        out[i..][0..W].* = op(c, ctx); // @Vector -> [W]T store coercion
    }
    if (rem != 0) {
        const c: @Vector(rem, T) = ba[full..][0..rem].*;
        out[full..][0..rem].* = op(c, ctx);
    }
    return out;
}

/// Binary version of `mapUnary`: `op(chunkA, chunkB, ctx) @TypeOf(chunkA)`.
pub fn mapBinary(a: anytype, b: anytype, ctx: anytype, comptime op: anytype) [meta.lengthOf(@TypeOf(a))]meta.Child(@TypeOf(a)) {
    const T = meta.Child(@TypeOf(a));
    const N = comptime meta.lengthOf(@TypeOf(a));
    const W = comptime laneCount(T);
    const full = comptime (N / W) * W;
    const rem = comptime N % W;

    const ba = a;
    const bb = b;
    var out: [N]T = undefined;
    var i: usize = 0;
    while (i < full) : (i += W) {
        const ca: @Vector(W, T) = ba[i..][0..W].*;
        const cb: @Vector(W, T) = bb[i..][0..W].*;
        out[i..][0..W].* = op(ca, cb, ctx);
    }
    if (rem != 0) {
        const ca: @Vector(rem, T) = ba[full..][0..rem].*;
        const cb: @Vector(rem, T) = bb[full..][0..rem].*;
        out[full..][0..rem].* = op(ca, cb, ctx);
    }
    return out;
}

/// Element-wise `a - b` returned as an ARRAY, so a following reduction chunks it
/// instead of materializing one wide `@Vector`. Used by distance / is_close.
pub fn sub(a: anytype, b: anytype) [meta.lengthOf(@TypeOf(a))]meta.Child(@TypeOf(a)) {
    return mapBinary(a, b, {}, struct {
        fn op(x: anytype, y: anytype, _: void) @TypeOf(x) {
            return x - y;
        }
    }.op);
}

/// Width-generic sin/cos core (cephes-derived). Operates on ONE `@Vector` of any
/// width and float element type. Called by both the `@Vector` public path and
/// the per-chunk array path so the math lives in exactly one place.
///
/// Implementation based on sinf.c from the cephes library, combining sinf and
/// cosf, changing octants to quadrants and vectorizing it.
/// Original implementation by Stephen L. Moshier (See: http://www.moshier.net/)
pub fn sinCosVec(in: anytype) struct { sin: @TypeOf(in), cos: @TypeOf(in) } {
    const FVec = @TypeOf(in);
    const info = switch (@typeInfo(FVec)) {
        .vector => |v| v,
        else => @compileError("sinCosVec expects a @Vector, got: " ++ @typeName(FVec)),
    };
    const w = info.len;
    const F = info.child;
    const bits = switch (@typeInfo(F)) {
        .float => |f| f.bits,
        else => @compileError("sinCosVec only supports floating point vectors"),
    };
    const UVec = @Vector(w, switch (bits) {
        32 => u32,
        64 => u64,
        else => @compileError("Unsupported float size"),
    });
    const last_bit = switch (bits) {
        32 => 0x80000000,
        64 => 0x8000000000000000,
        else => unreachable,
    };
    const num_bits = bits;

    // Make argument positive and remember sign for sin only since cos is symmetric around x (highest bit of a float is the sign bit)
    var sin_sign = @as(UVec, @bitCast(in)) & @as(UVec, @splat(last_bit));
    var x: FVec = @bitCast(@as(UVec, @bitCast(in)) ^ sin_sign);

    // x / (PI / 2) rounded to nearest int gives us the quadrant closest to x
    const quadrant: UVec = @intFromFloat(@as(FVec, @splat(0.6366197723675814)) * x + @as(FVec, @splat(0.5)));
    const float_quadrant: FVec = @floatFromInt(quadrant);

    // Make x relative to the closest quadrant using a two step Cody-Waite argument reduction.
    // After this we have x in the range [-PI / 4, PI / 4].
    x = ((x - float_quadrant * @as(FVec, @splat(1.5703125))) - float_quadrant * @as(FVec, @splat(0.0004837512969970703125))) - float_quadrant * @as(FVec, @splat(7.549789948768648e-8));

    // Calculate x2 = x^2
    const x2 = x * x;

    // Taylor expansion:
    // Cos(x) = 1 - x^2/2! + x^4/4! - x^6/6! + x^8/8! + ... = (((x2/8!- 1/6!) * x2 + 1/4!) * x2 - 1/2!) * x2 + 1
    const taylor_cos = ((@as(FVec, @splat(2.443315711809948e-5)) * x2 - @as(FVec, @splat(1.388731625493765e-3))) * x2 + @as(FVec, @splat(4.166664568298827e-2))) * x2 * x2 - @as(FVec, @splat(0.5)) * x2 + @as(FVec, @splat(1.0));
    // Sin(x) = x - x^3/3! + x^5/5! - x^7/7! + ... = ((-x2/7! + 1/5!) * x2 - 1/3!) * x2 * x + x
    const taylor_sin = ((@as(FVec, @splat(-1.9515295891e-4)) * x2 + @as(FVec, @splat(8.3321608736e-3))) * x2 - @as(FVec, @splat(1.6666654611e-1))) * x2 * x + x;

    // The lowest 2 bits of quadrant indicate the quadrant that we are in. Extract
    // them to determine which Taylor expansion to use and the signs.
    const bit1: UVec = quadrant << @as(UVec, @splat(num_bits - 1)); // bit 0
    const bit2: UVec = (quadrant << @as(UVec, @splat(num_bits - 2))) & @as(UVec, @splat(last_bit)); // bit 1

    // Select which one of the results is sin and which one is cos based on bit1
    const s = @select(F, bit1 > @as(UVec, @splat(0)), taylor_cos, taylor_sin);
    const c = @select(F, bit1 > @as(UVec, @splat(0)), taylor_sin, taylor_cos);

    sin_sign = sin_sign ^ bit2;
    const cos_sign = bit1 ^ bit2;

    return .{
        .sin = @as(FVec, @bitCast(@as(UVec, @bitCast(s)) ^ sin_sign)),
        .cos = @as(FVec, @bitCast(@as(UVec, @bitCast(c)) ^ cos_sign)),
    };
}

/// Chunked array driver for sin/cos: returns two `[N]T` arrays.
pub fn sinCos(a: anytype) struct {
    sin: [meta.lengthOf(@TypeOf(a))]meta.Child(@TypeOf(a)),
    cos: [meta.lengthOf(@TypeOf(a))]meta.Child(@TypeOf(a)),
} {
    const T = meta.Child(@TypeOf(a));
    const N = comptime meta.lengthOf(@TypeOf(a));
    const W = comptime laneCount(T);
    const full = comptime (N / W) * W;
    const rem = comptime N % W;

    const ba = a;
    var s: [N]T = undefined;
    var c: [N]T = undefined;
    var i: usize = 0;
    while (i < full) : (i += W) {
        const chunk: @Vector(W, T) = ba[i..][0..W].*;
        const r = sinCosVec(chunk);
        s[i..][0..W].* = r.sin;
        c[i..][0..W].* = r.cos;
    }
    if (rem != 0) {
        const chunk: @Vector(rem, T) = ba[full..][0..rem].*;
        const r = sinCosVec(chunk);
        s[full..][0..rem].* = r.sin;
        c[full..][0..rem].* = r.cos;
    }
    return .{ .sin = s, .cos = c };
}

test laneCount {
    // Always at least 1 lane so chunking is well-defined on any target.
    try std.testing.expect(laneCount(f32) >= 1);
    try std.testing.expect(laneCount(i8) >= 1);
}

test sumSquares {
    // Chunked reduction over a remainder-forcing length.
    const a = [_]f32{ 1, 2, 3, 4, 5, 6, 7 };
    try std.testing.expectApproxEqAbs(@as(f32, 140), sumSquares(f32, a), 1e-3);
    // Integer accumulation is exact and associative.
    const ai = [_]i32{ 1, 2, 3, 4, 5 };
    try std.testing.expectEqual(@as(i32, 55), sumSquares(i32, ai));
}

test sumProducts {
    const a = [_]f32{ 1, 2, 3, 4, 5 };
    const b = [_]f32{ 5, 4, 3, 2, 1 };
    try std.testing.expectApproxEqAbs(@as(f32, 35), sumProducts(f32, a, b), 1e-3);
}

test sub {
    const a = [_]f32{ 4, 6, 8 };
    const b = [_]f32{ 1, 2, 3 };
    const d = sub(a, b);
    try std.testing.expect(@TypeOf(d) == [3]f32);
    try std.testing.expectEqual([3]f32{ 3, 4, 5 }, d);
}

test mapUnary {
    const a = [_]f32{ 1, 2, 3, 4, 5 };
    const out = mapUnary(a, @as(f32, 2), struct {
        fn op(c: anytype, s: f32) @TypeOf(c) {
            return c * @as(@TypeOf(c), @splat(s));
        }
    }.op);
    try std.testing.expectEqual([5]f32{ 2, 4, 6, 8, 10 }, out);
}
