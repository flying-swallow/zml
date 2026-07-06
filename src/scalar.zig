const std = @import("std");

/// Scalar element type of a scalar-or-vector type (`f32` -> `f32`, `@Vector(n, f32)` -> `f32`).
fn Elem(comptime V: type) type {
    return switch (@typeInfo(V)) {
        .vector => |info| info.child,
        else => V,
    };
}

/// Broadcast a scalar to `V` (splat for vectors, identity for scalars).
fn splatLike(comptime V: type, value: Elem(V)) V {
    return switch (@typeInfo(V)) {
        .vector => @splat(value),
        else => value,
    };
}

/// Clamp `x` to the inclusive range [lo, hi]. Works on scalars and float vectors.
pub fn clamp(x: anytype, lo: Elem(@TypeOf(x)), hi: Elem(@TypeOf(x))) @TypeOf(x) {
    const V = @TypeOf(x);
    return @min(@max(x, splatLike(V, lo)), splatLike(V, hi));
}

/// Clamp `x` to [0, 1].
pub fn saturate(x: anytype) @TypeOf(x) {
    return clamp(x, 0, 1);
}

/// Linear interpolation: a + (b - a) * t.
pub fn lerp(a: anytype, b: @TypeOf(a), t: Elem(@TypeOf(a))) @TypeOf(a) {
    const V = @TypeOf(a);
    return a + (b - a) * splatLike(V, t);
}

/// Multiply-add: a * b + c.
pub fn madd(a: anytype, b: @TypeOf(a), c: @TypeOf(a)) @TypeOf(a) {
    return a * b + c;
}

/// 0 if x < edge, else 1 (component-wise). `edge` and `x` share the same type.
pub fn step(edge: anytype, x: @TypeOf(edge)) @TypeOf(edge) {
    const V = @TypeOf(edge);
    return switch (@typeInfo(V)) {
        .vector => @select(Elem(V), x >= edge, splatLike(V, 1), splatLike(V, 0)),
        else => @as(V, if (x >= edge) 1 else 0),
    };
}

/// Saturated linear ramp from edge0 to edge1.
pub fn linearstep(edge0: anytype, edge1: @TypeOf(edge0), x: @TypeOf(edge0)) @TypeOf(edge0) {
    return saturate((x - edge0) / (edge1 - edge0));
}

/// Hermite smoothstep: 0 below edge0, 1 above edge1, smooth in between.
pub fn smoothstep(edge0: anytype, edge1: @TypeOf(edge0), x: @TypeOf(edge0)) @TypeOf(edge0) {
    const V = @TypeOf(edge0);
    const t = saturate((x - edge0) / (edge1 - edge0));
    return t * t * (splatLike(V, 3) - splatLike(V, 2) * t);
}

test "scalar helpers" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), lerp(@as(f32, 0), @as(f32, 1), 0.5), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 7), madd(@as(f32, 2), @as(f32, 3), @as(f32, 1)), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), saturate(@as(f32, -3)), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), saturate(@as(f32, 3)), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.75), clamp(@as(f32, 5), 0.25, 0.75), 1e-6);

    try std.testing.expectApproxEqAbs(@as(f32, 0), smoothstep(@as(f32, 0), @as(f32, 1), @as(f32, 0)), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), smoothstep(@as(f32, 0), @as(f32, 1), @as(f32, 1)), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), smoothstep(@as(f32, 0), @as(f32, 1), @as(f32, 0.5)), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), linearstep(@as(f32, 0), @as(f32, 4), @as(f32, 1)), 1e-6);

    try std.testing.expectApproxEqAbs(@as(f32, 1), step(@as(f32, 0.5), @as(f32, 0.7)), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), step(@as(f32, 0.5), @as(f32, 0.2)), 1e-6);

    // Vector variants.
    try std.testing.expectEqual(@Vector(2, f32){ 1, 2 }, lerp(@Vector(2, f32){ 0, 0 }, @Vector(2, f32){ 2, 4 }, 0.5));
    try std.testing.expectEqual(@Vector(2, f32){ 0, 1 }, saturate(@Vector(2, f32){ -1, 2 }));
    try std.testing.expectEqual(@Vector(2, f32){ 0, 1 }, step(@Vector(2, f32){ 0.5, 0.5 }, @Vector(2, f32){ 0.2, 0.7 }));
}
