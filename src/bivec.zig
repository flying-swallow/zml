//! Bivectors: the logarithms of rotors, i.e. oriented areas / planes of rotation. A rotor's
//! angular velocity is a bivector.
//!
//! Representation (bare `@Vector`, so `Bivec2f32 == Vec2f32`-shaped is accepted):
//!   * `Bivec2f32 = @Vector(1, f32)` -- component `[0] = yx` (the only plane in 2d).
//!   * `Bivec3f32 = @Vector(3, f32)` -- components `[0] = yz`, `[1] = xz`, `[2] = yx`.
//!
//! Keeping 2d as a 1-vector (rather than a bare `f32`) means `@splat` works uniformly and the
//! dimension stays recoverable from the type. Ported from Games-by-Mason's mr_geom (MIT).

const std = @import("std");
const vector = @import("vector.zig");
const meta = @import("meta.zig");

pub const Bivec2f32 = @Vector(1, f32);
pub const Bivec3f32 = @Vector(3, f32);

/// The zero bivector (no rotation), typed as `B`.
pub inline fn zero(comptime B: type) B {
    return @splat(0);
}

/// The unit bivector on the yx plane. 2d: `{1}`. 3d: `{0, 0, 1}`.
pub inline fn yx_plane(comptime B: type) B {
    var result: B = @splat(0);
    result[meta.lengthOf(B) - 1] = 1;
    return result;
}

/// The unit bivector on the xy plane (opposite orientation of `yx_plane`).
pub inline fn xy_plane(comptime B: type) B {
    var result: B = @splat(0);
    result[meta.lengthOf(B) - 1] = -1;
    return result;
}

/// The bivector scaled by `factor`.
pub inline fn scaled(b: anytype, factor: anytype) @TypeOf(b) {
    const V = meta.AsVector(@TypeOf(b));
    return @as(V, b) * @as(V, @splat(factor));
}

/// Squared magnitude.
pub inline fn mag_sq(b: anytype) meta.Child(@TypeOf(b)) {
    return vector.norm_sqr(b);
}

/// Magnitude.
pub inline fn mag(b: anytype) meta.Child(@TypeOf(b)) {
    return vector.norm(b);
}

/// Compare bivector magnitudes without a square root. `mag_cmp(b)` and `mag_cmp_to(m)` form a
/// pair: `mag_cmp(b) <=> mag_cmp_to(m)` iff `mag(b) <=> m`, for `m >= 0`. Unified across
/// dimensions as `(norm_sqr, m*m)` -- both satisfy the contract, and the shared form removes the
/// dimension switch mr_geom needed.
pub inline fn mag_cmp(b: anytype) meta.Child(@TypeOf(b)) {
    return vector.norm_sqr(b);
}

/// See `mag_cmp`. Converts a magnitude threshold into comparison space.
pub inline fn mag_cmp_to(m: anytype) @TypeOf(m) {
    return m * m;
}

test "planes and zero" {
    try std.testing.expectEqual(Bivec2f32{0}, zero(Bivec2f32));
    try std.testing.expectEqual(Bivec2f32{1}, yx_plane(Bivec2f32));
    try std.testing.expectEqual(Bivec2f32{-1}, xy_plane(Bivec2f32));
    try std.testing.expectEqual(Bivec3f32{ 0, 0, 0 }, zero(Bivec3f32));
    try std.testing.expectEqual(Bivec3f32{ 0, 0, 1 }, yx_plane(Bivec3f32));
    try std.testing.expectEqual(Bivec3f32{ 0, 0, -1 }, xy_plane(Bivec3f32));
}

test "mag_cmp pair contract holds in both dimensions" {
    // 2d
    {
        const b: Bivec2f32 = .{2};
        try std.testing.expect(mag_cmp(b) > mag_cmp_to(@as(f32, 1))); // |2| > 1
        try std.testing.expect(mag_cmp(b) < mag_cmp_to(@as(f32, 3))); // |2| < 3
    }
    // 3d
    {
        const b: Bivec3f32 = .{ 0, 0, 2 };
        try std.testing.expect(mag_cmp(b) > mag_cmp_to(@as(f32, 1)));
        try std.testing.expect(mag_cmp(b) < mag_cmp_to(@as(f32, 3)));
    }
}

test "scaled" {
    try std.testing.expectEqual(Bivec2f32{2}, scaled(Bivec2f32{1}, @as(f32, 2)));
    try std.testing.expectEqual(Bivec3f32{ 2, 4, 6 }, scaled(Bivec3f32{ 1, 2, 3 }, @as(f32, 2)));
}
