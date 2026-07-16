//! Fast approximations and endpoint-exact helpers used by the rotor and spring math.
//! Ported from Games-by-Mason's mr_geom and mr_tween (MIT).

const std = @import("std");

/// Very fast approximate inverse square root that only produces usable results when `f` is near
/// one. Converges for `f` in `(0, ~1.5615528128088303)`. Used to renormalize rotors that are
/// already close to unit length.
pub fn inv_sqrt_near_one(f: anytype) @TypeOf(f) {
    return @mulAdd(@TypeOf(f), f, -0.5, 1.5);
}

test inv_sqrt_near_one {
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), inv_sqrt_near_one(@as(f32, 1.0)), 1e-6);
    // First-order approximation near 1: error grows with |f - 1|, so give it room at f = 1.1.
    try std.testing.expectApproxEqAbs(1.0 / @sqrt(@as(f32, 1.1)), inv_sqrt_near_one(@as(f32, 1.1)), 5e-3);
}

/// Fast approximate inverse square root using a hardware instruction when available.
pub fn inv_sqrt(f: anytype) @TypeOf(f) {
    // Positive infinity -> infinity.
    if (std.math.isPositiveInf(f)) return f;
    // NaN, negative infinity, or zero -> NaN.
    if (std.math.isNan(f) or std.math.isNegativeInf(f) or f == 0.0) {
        return std.math.nan(@TypeOf(f));
    }
    // With the degenerate inputs ruled out, enable fast-math so this lowers to a hardware
    // rsqrt on platforms that have one. Dropping `.optimized` would silently regress perf.
    {
        @setFloatMode(.optimized);
        return 1.0 / @sqrt(f);
    }
}

test inv_sqrt {
    try std.testing.expectApproxEqAbs(1.0 / @sqrt(@as(f32, 4.0)), inv_sqrt(@as(f32, 4.0)), 1e-4);
    try std.testing.expect(std.math.isNan(inv_sqrt(@as(f32, 0.0))));
    try std.testing.expect(std.math.isPositiveInf(inv_sqrt(std.math.inf(f32))));
}

/// Approximation of `exp(-x)` for `x >= 0`. Used by the damped spring integrator.
pub fn inv_exp(x: anytype) @TypeOf(x) {
    const T = @TypeOf(x);
    const c0 = 1.119186;
    const c1 = 0.097090;
    const c2 = 0.489114;
    const xx = x * x;
    const xxx = xx * x;
    return 1.0 / (@mulAdd(T, c0, x, 1) + @mulAdd(T, c1, xx, c2 * xxx));
}

test inv_exp {
    try std.testing.expectApproxEqAbs(@exp(-@as(f32, 0.5)), inv_exp(@as(f32, 0.5)), 0.01);
    try std.testing.expectApproxEqAbs(@exp(-@as(f32, 2.0)), inv_exp(@as(f32, 2.0)), 0.01);
}

/// Approximation of `exp(-x)` tuned for `x` in `[0, 1]`.
pub fn inv_exp_01(x: anytype) @TypeOf(x) {
    const T = @TypeOf(x);
    const c0 = 1.006722;
    const c1 = 0.453568;
    const c2 = 0.254559;
    const xx = x * x;
    const xxx = xx * x;
    return 1.0 / (@mulAdd(T, c0, x, 1) + @mulAdd(T, c1, xx, c2 * xxx));
}

test inv_exp_01 {
    try std.testing.expectApproxEqAbs(@exp(-@as(f32, 0.25)), inv_exp_01(@as(f32, 0.25)), 0.01);
}

/// `log2(x)` with a linear approximation within `eps` of one (avoids catastrophic cancellation).
pub fn log2_near_one(x: anytype, eps: @TypeOf(x)) @TypeOf(x) {
    const x_minus_1 = x - 1;
    if (@abs(x_minus_1) > eps) {
        return @log2(x);
    } else {
        return x_minus_1 / @log(2.0);
    }
}

test log2_near_one {
    const eps = std.math.floatEps(f32);
    try std.testing.expectEqual(@as(f32, 0), log2_near_one(@as(f32, 1), eps));
    try std.testing.expectApproxEqAbs(@log2(@as(f32, 1.5)), log2_near_one(@as(f32, 1.5), eps), 1e-6);
}

/// `exp2(x) = 2^x` with a linear approximation within `eps` of zero.
pub fn exp2_near_zero(x: anytype, eps: @TypeOf(x)) @TypeOf(x) {
    if (@abs(x) > eps) {
        return @exp2(x);
    } else {
        return x * @log(2.0) + 1;
    }
}

test exp2_near_zero {
    const eps = std.math.floatEps(f32);
    try std.testing.expectEqual(@as(f32, 1), exp2_near_zero(@as(f32, 0), eps));
    try std.testing.expectApproxEqAbs(@exp2(@as(f32, 0.5)), exp2_near_zero(@as(f32, 0.5), eps), 1e-6);
}

/// Linear interpolation that is exact at both endpoints (unlike `a + (b - a) * t`, which is not
/// exact at `t == 1`). `nlerp` and the spring math rely on the endpoint-exact property.
pub fn lerp_exact(start: anytype, end: @TypeOf(start), t: anytype) @TypeOf(start) {
    const T = @TypeOf(start);
    return @mulAdd(T, start, splat(T, 1.0 - t), end * splat(T, t));
}

/// `@splat`-with-scalar-passthrough: broadcasts `s` to a vector `T`, or returns it as-is for
/// scalar `T`.
pub inline fn splat(comptime T: type, s: anytype) T {
    return switch (@typeInfo(T)) {
        .vector => @splat(s),
        else => s,
    };
}

test lerp_exact {
    try std.testing.expectEqual(@as(f32, 3), lerp_exact(@as(f32, 1), @as(f32, 5), @as(f32, 0.5)));
    // Exact at both ends.
    try std.testing.expectEqual(@as(f32, 5), lerp_exact(@as(f32, 1), @as(f32, 5), @as(f32, 1.0)));
    try std.testing.expectEqual(@as(f32, 1), lerp_exact(@as(f32, 1), @as(f32, 5), @as(f32, 0.0)));
}
