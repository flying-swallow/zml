//! Rotors: a generalization of quaternions that works in any dimension. A rotor is the even
//! subalgebra of the geometric algebra; `exp`/`ln` bridge it to a `bivec` (its rotation plane
//! scaled by half the angle). Ported from Games-by-Mason's mr_geom (MIT).
//!
//! ONE module handles both 2d and 3d: the dimension is recoverable from the lane count, so every
//! function dispatches on `meta.lengthOf` rather than duplicating a `rotor2`/`rotor3` split.
//!
//! Representation (bare `@Vector`, so `Rotor2f32 == Vec2f32`-shaped is accepted). Array inputs are
//! also accepted and the container kind is preserved on the way out (`[N]f32` in -> `[N]f32` out),
//! which is what lets mr_ecs store the compact `[N]f32` form:
//!   * `Rotor2f32 = @Vector(2, f32)` -- lanes `[0] = yx` (sin of half angle), `[1] = a` (cos).
//!   * `Rotor3f32 = @Vector(4, f32)` -- lanes `[0] = yz`, `[1] = xz`, `[2] = yx`, `[3] = a`.
//!
//! ⚠️ `Rotor3f32` is byte-identical to `Quat4f32` but the lanes differ (`xz`, not `zx`), so
//! `quat.mul(rotor_a, rotor_b)` compiles, runs, and is silently WRONG. The final test in this file
//! pins that they disagree. Use `rotor.times` for rotors.

const std = @import("std");
const meta = @import("meta.zig");
const bivec = @import("bivec.zig");
const approx = @import("approx.zig");
const quat = @import("quat.zig");

pub const Rotor2f32 = @Vector(2, f32);
pub const Rotor3f32 = @Vector(4, f32);

/// The bivector (log) type paired with a rotor type `R`, preserving `R`'s container kind.
pub fn BivecForRotor(comptime R: type) type {
    return meta.Reshape(R, if (meta.lengthOf(R) == 2) 1 else 3);
}

/// The rotor (exp) type paired with a bivector type `B`, preserving `B`'s container kind.
pub fn RotorForBivec(comptime B: type) type {
    return meta.Reshape(B, if (meta.lengthOf(B) == 1) 2 else 4);
}

/// The rotor type produced by `from_to` on a length-N vector type `Vt`.
pub fn RotorForVec(comptime Vt: type) type {
    return meta.Reshape(Vt, if (meta.lengthOf(Vt) == 2) 2 else 4);
}

// The threshold structs below are unified across dimensions: everything compares SQUARED
// magnitudes, so a single dimension-independent struct replaces mr_geom's separate 2d/3d options.
// `x*x <=> m*m  <=>  |x| <=> m` for m >= 0, and the squared eps values equal mr_geom's squared 3d
// defaults, so the branch taken (and thus the result) matches on both the 2d and 3d paths.

pub const LnEpsOptions = struct {
    /// Disable approximations.
    pub const precise: @This() = .{ .eps = 0 };
    /// If the squared sine of the half angle is <= this, a faster approximation is used. Defaults
    /// to roughly less than a degree of round-trip error.
    eps: f32 = std.math.pow(f32, @sin(0.75 / 2.0), 2.0),
};

pub const ExpEpsOptions = struct {
    /// Disable approximations.
    pub const precise: @This() = .{ .eps = 0 };
    /// If the squared half angle is <= this, a faster approximation is used. Defaults to roughly
    /// less than a degree of round-trip error.
    eps: f32 = std.math.pow(f32, 0.75 / 2.0, 2.0),
};

pub const SlerpEpsOptions = struct {
    /// Disable the approximations.
    pub const precise: @This() = .{ .ln = .precise, .exp = .precise };
    ln: LnEpsOptions = .{},
    exp: ExpEpsOptions = .{},
};

/// The identity rotor (no rotation), typed as `R`. Valid as a comptime field default.
pub inline fn identity(comptime R: type) R {
    var v: meta.AsVector(R) = @splat(0);
    v[meta.lengthOf(R) - 1] = 1;
    return v;
}

/// Returns the rotor with every component scaled by `factor`. Does not scale the rotation.
pub inline fn scaled_comps(self: anytype, factor: meta.Child(@TypeOf(self))) @TypeOf(self) {
    const V = meta.AsVector(@TypeOf(self));
    return @as(V, self) * @as(V, @splat(factor));
}

/// Squared magnitude of the rotor. Matches mr_geom's `@mulAdd` grouping exactly.
pub inline fn mag_sq(self: anytype) meta.Child(@TypeOf(self)) {
    const T = meta.Child(@TypeOf(self));
    const v: meta.AsVector(@TypeOf(self)) = self;
    if (meta.lengthOf(@TypeOf(self)) == 2) {
        return @mulAdd(T, v[0], v[0], v[1] * v[1]);
    } else {
        return @mulAdd(T, v[0], v[0], v[1] * v[1]) + @mulAdd(T, v[2], v[2], v[3] * v[3]);
    }
}

/// Magnitude of the rotor. Prefer `mag_sq` with `approx.inv_sqrt`/`inv_sqrt_near_one`.
pub inline fn mag(self: anytype) meta.Child(@TypeOf(self)) {
    return @sqrt(mag_sq(self));
}

/// The inverse rotation (negate the bivector part). Not the same as `negated`.
pub inline fn inverse(self: anytype) @TypeOf(self) {
    const V = meta.AsVector(@TypeOf(self));
    const v: V = self;
    if (meta.lengthOf(@TypeOf(self)) == 2) {
        return V{ -v[0], v[1] };
    } else {
        return V{ -v[0], -v[1], -v[2], v[3] };
    }
}

/// Cosine of the half angle between two normalized rotors. Negative => more than a full rotation
/// apart. Matches mr_geom's `@mulAdd` grouping exactly.
pub inline fn neighborhood(self: anytype, other: @TypeOf(self)) meta.Child(@TypeOf(self)) {
    const T = meta.Child(@TypeOf(self));
    const a: meta.AsVector(@TypeOf(self)) = self;
    const b: meta.AsVector(@TypeOf(self)) = other;
    if (meta.lengthOf(@TypeOf(self)) == 2) {
        return @mulAdd(T, a[0], b[0], a[1] * b[1]);
    } else {
        return @mulAdd(T, a[0], b[0], @mulAdd(T, a[1], b[1], @mulAdd(T, a[2], b[2], a[3] * b[3])));
    }
}

/// Renormalizes a rotor assumed already near unit length (fast, pure `@mulAdd`).
pub inline fn renormalized(self: anytype) @TypeOf(self) {
    return scaled_comps(self, approx.inv_sqrt_near_one(mag_sq(self)));
}

/// Normalizes a rotor. NaN-filled if the magnitude is 0. Reaches `approx.inv_sqrt` (fast-math).
pub inline fn normalized(self: anytype) @TypeOf(self) {
    return scaled_comps(self, approx.inv_sqrt(mag_sq(self)));
}

/// Creates a 2d rotor from an angle in radians. Prefer `from_to`/`from_scaled_plane` for non-human
/// input. Angles greater than 2*pi wrap.
pub inline fn from_angle(comptime R: type, rad: meta.Child(R)) R {
    comptime std.debug.assert(meta.lengthOf(R) == 2);
    const half = rad * 0.5;
    return meta.AsVector(R){ @sin(half), @cos(half) };
}

/// Creates a rotor from a unit `plane` and an angle in radians. Works in 2d (single plane) and 3d.
pub inline fn from_plane_angle(plane: anytype, rad: meta.Child(@TypeOf(plane))) RotorForBivec(@TypeOf(plane)) {
    const B = @TypeOf(plane);
    const RV = meta.AsVector(RotorForBivec(B));
    const bv: meta.AsVector(B) = plane;
    const half = rad * 0.5;
    const sin = @sin(half);
    const cos = @cos(half);
    if (meta.lengthOf(B) == 1) {
        return RV{ sin * bv[0], cos };
    } else {
        return RV{ sin * bv[0], sin * bv[1], sin * bv[2], cos };
    }
}

/// Composes two rotations: applies `other`, then `self`. Order matters (non-commutative in 3d).
pub inline fn times(self: anytype, other: @TypeOf(self)) @TypeOf(self) {
    const T = meta.Child(@TypeOf(self));
    const V = meta.AsVector(@TypeOf(self));
    const a: V = self;
    const b: V = other;
    if (meta.lengthOf(@TypeOf(self)) == 2) {
        const result = V{
            @mulAdd(T, a[0], b[1], a[1] * b[0]), // yx
            @mulAdd(T, -a[0], b[0], a[1] * b[1]), // a
        };
        return renormalized(result);
    } else {
        const result = V{
            a[0] * b[3] - a[1] * b[2] + a[2] * b[1] + a[3] * b[0], // yz
            a[0] * b[2] + a[1] * b[3] - a[2] * b[0] + a[3] * b[1], // xz
            -a[0] * b[1] + a[1] * b[0] + a[2] * b[3] + a[3] * b[2], // yx
            -a[0] * b[0] - a[1] * b[1] - a[2] * b[2] + a[3] * b[3], // a
        };
        return renormalized(result);
    }
}

/// Applies the rotor to a vector (the sandwich product), rotating it.
pub inline fn times_vec(self: anytype, point: anytype) @TypeOf(point) {
    const T = meta.Child(@TypeOf(self));
    const s: meta.AsVector(@TypeOf(self)) = self;
    const PV = meta.AsVector(@TypeOf(point));
    const p: PV = point;
    if (meta.lengthOf(@TypeOf(self)) == 2) {
        // s[0]=yx, s[1]=a ; p[0]=x, p[1]=y
        const x = @mulAdd(T, s[1], p[0], s[0] * p[1]);
        const y = @mulAdd(T, s[1], p[1], -s[0] * p[0]);
        return PV{
            @mulAdd(T, x, s[1], y * s[0]),
            @mulAdd(T, x, -s[0], y * s[1]),
        };
    } else {
        // s: yz,xz,yx,a ; p: x,y,z
        const x = @mulAdd(T, -s[1], p[2], @mulAdd(T, s[2], p[1], s[3] * p[0]));
        const y = @mulAdd(T, -s[0], p[2], @mulAdd(T, -s[2], p[0], s[3] * p[1]));
        const z = @mulAdd(T, s[0], p[1], @mulAdd(T, s[1], p[0], s[3] * p[2]));
        const xyz = @mulAdd(T, -s[0], p[0], @mulAdd(T, s[1], p[1], s[2] * p[2]));
        return PV{
            y * s[2] - z * s[1] - xyz * s[0] + x * s[3],
            -x * s[2] - z * s[0] + xyz * s[1] + y * s[3],
            x * s[1] + y * s[0] + xyz * s[2] + z * s[3],
        };
    }
}

/// Creates a rotor that rotates along the shortest path from `from` to `to`, both normalized. If
/// the vectors are antiparallel the tie is broken arbitrarily; if either is zero, identity. Reaches
/// `approx.inv_sqrt` (fast-math).
pub inline fn from_to(from: anytype, to: @TypeOf(from)) RotorForVec(@TypeOf(from)) {
    const T = meta.Child(@TypeOf(from));
    const RV = meta.AsVector(RotorForVec(@TypeOf(from)));
    const f: meta.AsVector(@TypeOf(from)) = from;
    const t: meta.AsVector(@TypeOf(from)) = to;
    if (meta.lengthOf(@TypeOf(from)) == 2) {
        const outer_yx = @mulAdd(T, -f[0], t[1], t[0] * f[1]);
        const inner = @mulAdd(T, f[0], t[0], f[1] * t[1]);
        const result = RV{ outer_yx, inner + 1.0 };
        const res_mag_sq = mag_sq(result);
        if (res_mag_sq == 0.0) return RV{ 1, 0 }; // 180 degrees, sign arbitrary
        return scaled_comps(result, approx.inv_sqrt(res_mag_sq));
    } else {
        const outer_yz = @mulAdd(T, f[1], t[2], -f[2] * t[1]);
        const outer_xz = @mulAdd(T, -f[2], t[0], f[0] * t[2]);
        const outer_yx = @mulAdd(T, -f[0], t[1], f[1] * t[0]);
        const inner = @mulAdd(T, f[0], t[0], @mulAdd(T, f[1], t[1], f[2] * t[2]));
        const result = RV{ outer_yz, outer_xz, outer_yx, inner + 1.0 };
        const res_mag_sq = mag_sq(result);
        if (res_mag_sq == 0.0) {
            // Antiparallel: rotate 180 degrees around any perpendicular plane.
            const BV = @Vector(3, T);
            // outerProd(from, x_pos) then outerProd(from, y_pos) if that is zero.
            var plane = BV{ 0, -f[2], f[1] };
            if (@mulAdd(T, plane[0], plane[0], @mulAdd(T, plane[1], plane[1], plane[2] * plane[2])) == 0) {
                plane = BV{ -f[2], 0, -f[0] };
            }
            return from_plane_angle(plane, std.math.pi);
        }
        return scaled_comps(result, approx.inv_sqrt(res_mag_sq));
    }
}

/// Natural log of a normalized rotor -> the bivector whose plane is the rotation plane and whose
/// magnitude is half the rotation angle. Uses a fast approximation at/below `options.eps`.
pub inline fn ln_eps(self: anytype, options: LnEpsOptions) BivecForRotor(@TypeOf(self)) {
    const T = meta.Child(@TypeOf(self));
    const B = BivecForRotor(@TypeOf(self));
    const BV = meta.AsVector(B);
    const s: meta.AsVector(@TypeOf(self)) = self;
    if (meta.lengthOf(@TypeOf(self)) == 2) {
        const yx = s[0];
        const cos = s[1];
        const sin_sq = yx * yx;
        // Below eps and in the identity's neighborhood: return the bivector part directly.
        if (sin_sq <= @max(options.eps, 0) and cos > 0) return BV{yx};
        return BV{std.math.atan2(yx, cos)};
    } else {
        const bivec_part = BV{ s[0], s[1], s[2] };
        const cos = s[3];
        const sin_sq = @mulAdd(T, s[0], s[0], @mulAdd(T, s[1], s[1], s[2] * s[2]));
        if (sin_sq <= @max(options.eps, 0) and cos > 0) return bivec_part;
        // sin == 0 with cos ruled positive => cos == -1: a 360-degree turn on an arbitrary plane.
        if (sin_sq == 0.0) return BV{ 0, 0, std.math.pi };
        const sin = @sqrt(sin_sq);
        const half = std.math.atan2(sin, cos);
        return bivec.scaled(bivec_part, half / sin);
    }
}

/// Raises e to a bivector -> a rotor that rotates on the bivector's plane by twice its magnitude.
/// Uses a fast approximation at/below `options.eps`.
pub inline fn exp_eps(self: anytype, options: ExpEpsOptions) RotorForBivec(@TypeOf(self)) {
    const T = meta.Child(@TypeOf(self));
    const R = RotorForBivec(@TypeOf(self));
    const RV = meta.AsVector(R);
    const b: meta.AsVector(@TypeOf(self)) = self;
    if (meta.lengthOf(@TypeOf(self)) == 1) {
        const yx = b[0];
        if (yx == 0) return identity(R);
        const m = @abs(yx);
        if (m * m <= options.eps) {
            const ms = @mulAdd(T, m, m, 1);
            return scaled_comps(RV{ yx, 1 }, approx.inv_sqrt_near_one(ms));
        }
        return RV{ @sin(yx), @cos(yx) };
    } else {
        const half_sq = @mulAdd(T, b[0], b[0], @mulAdd(T, b[1], b[1], b[2] * b[2]));
        if (half_sq == 0) return identity(R);
        if (half_sq <= options.eps) {
            const ms = half_sq + 1;
            return scaled_comps(RV{ b[0], b[1], b[2], 1 }, approx.inv_sqrt_near_one(ms));
        }
        const half = @sqrt(half_sq);
        const sin_half = @sin(half);
        const cos_half = @cos(half);
        const sin_half_scaled = sin_half / half;
        return RV{ sin_half_scaled * b[0], sin_half_scaled * b[1], sin_half_scaled * b[2], cos_half };
    }
}

/// The rotation plane scaled by the full rotation angle in radians (== `ln * 2`).
pub inline fn to_scaled_plane_eps(self: anytype, options: LnEpsOptions) BivecForRotor(@TypeOf(self)) {
    return bivec.scaled(ln_eps(self, options), @as(meta.Child(@TypeOf(self)), 2.0));
}

/// Inverse of `to_scaled_plane_eps`: a rotor from a plane scaled by its full angle.
pub inline fn from_scaled_plane_eps(plane: anytype, options: ExpEpsOptions) RotorForBivec(@TypeOf(plane)) {
    return exp_eps(bivec.scaled(plane, @as(meta.Child(@TypeOf(plane)), 0.5)), options);
}

/// Linearly interpolates then renormalizes. Cheaper and commutative vs slerp; well-behaved on
/// [0, 1]. Reaches `approx.inv_sqrt` (fast-math). Exact at both endpoints.
pub inline fn nlerp(start: anytype, end: @TypeOf(start), t: meta.Child(@TypeOf(start))) @TypeOf(start) {
    const V = meta.AsVector(@TypeOf(start));
    const result: V = approx.lerp_exact(@as(V, start), @as(V, end), t);
    const res_mag_sq = mag_sq(result);
    if (res_mag_sq == 0.0) return start;
    return scaled_comps(result, approx.inv_sqrt(res_mag_sq));
}

/// Options for `apply_spring`, parameterized on the rotor type `R`.
pub fn ApplySpringOptions(comptime R: type) type {
    return struct {
        /// The first derivative of the rotation (a bivector: plane = rotation plane, magnitude =
        /// radians per unit time).
        d1: *BivecForRotor(R),
        /// The target orientation.
        target: R,
        /// Approximation thresholds.
        eps: SlerpEpsOptions = .{},
    };
}

/// Applies a damped-spring matrix to the rotor pointed to by `self`, driving it toward
/// `options.target` and integrating `options.d1`. Exact at dt=0; exact at inf with nonzero damping.
///
/// The bivector state is bridged through a stack array so it hits `spring.apply_origin`'s
/// byte-identical `.array` branch (the generic `.vector` branch does not compile on Zig 0.17).
pub inline fn apply_spring(
    self: anytype,
    spring: anytype,
    options: ApplySpringOptions(@TypeOf(self.*)),
) void {
    const R = @TypeOf(self.*);
    const T = meta.Child(R);
    const RV = meta.AsVector(R);
    const BV = @Vector(meta.lengthOf(BivecForRotor(R)), T);
    const blen = meta.lengthOf(BV);

    // Exact at dt=0.
    if (std.meta.eql(spring, @TypeOf(spring).identity(0))) return;

    const cur: RV = self.*;
    const tgt: RV = options.target;
    const dist: BV = to_scaled_plane_eps(times(cur, inverse(tgt)), options.eps.ln);

    var d0_arr: [blen]T = dist;
    var d1_arr: [blen]T = options.d1.*;
    spring.apply_origin([blen]T, .{ .d0 = &d0_arr, .d1 = &d1_arr });
    options.d1.* = d1_arr;

    const new_rot: RV = times(from_scaled_plane_eps(@as(BV, d0_arr), options.eps.exp), tgt);
    self.* = new_rot;
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

const spring_mod = @import("spring.zig");
const zml_testing = @import("testing.zig");

fn expectRotorClose(expected: anytype, actual: @TypeOf(expected), tol: f32) !void {
    try zml_testing.expect_approx_eq_abs(@as(@Vector(meta.lengthOf(@TypeOf(expected)), f32), expected), actual, tol);
}

test "identity" {
    try std.testing.expectEqual(Rotor2f32{ 0, 1 }, identity(Rotor2f32));
    try std.testing.expectEqual(Rotor3f32{ 0, 0, 0, 1 }, identity(Rotor3f32));
    // Container kind preserved / valid as a comptime constant.
    try std.testing.expectEqual([2]f32{ 0, 1 }, comptime identity([2]f32));
}

test "from_angle and times_vec rotate a 2d vector" {
    const r = from_angle(Rotor2f32, std.math.pi / 2.0);
    // 90 degrees on the yx plane sends +x to -y.
    try expectRotorClose(@Vector(2, f32){ 0, -1 }, times_vec(r, @Vector(2, f32){ 1, 0 }), 1e-6);
}

test "times composition is non-commutative in 3d (order landmine guard)" {
    const a = from_plane_angle(bivec.Bivec3f32{ 0, 0, 1 }, std.math.pi / 2.0); // yx plane
    const b = from_plane_angle(bivec.Bivec3f32{ 1, 0, 0 }, std.math.pi / 2.0); // yz plane
    // These two orders must differ, or the composition-order test proves nothing.
    try std.testing.expect(!std.meta.eql(times(a, b), times(b, a)));
}

test "inverse composes to identity" {
    const r = from_plane_angle(bivec.Bivec3f32{ 1, 2, 3 }, 0.7);
    const rn = normalized(r);
    try expectRotorClose(identity(Rotor3f32), times(rn, inverse(rn)), 1e-5);
}

test "ln/exp round trip (2d and 3d)" {
    // Precise options make the round trip exact (the default eps enables a deliberately-approximate
    // small-angle path -- see the separate approximate-round-trip test below).
    {
        const r = from_angle(Rotor2f32, 0.6);
        const b = ln_eps(r, .precise);
        try expectRotorClose(r, exp_eps(b, .precise), 1e-5);
    }
    {
        const r = normalized(from_plane_angle(bivec.Bivec3f32{ 1, 2, 3 }, 0.6));
        const b = ln_eps(r, .precise);
        try expectRotorClose(r, exp_eps(b, .precise), 1e-5);
    }
}

test "ln/exp approximate round trip stays within ~1 degree" {
    // The default eps enables the fast small-angle approximation; it must still round trip closely.
    inline for (.{ 0.05, 0.2, 0.35 }) |ang| {
        const r2 = from_angle(Rotor2f32, ang);
        try expectRotorClose(r2, exp_eps(ln_eps(r2, .{}), .{}), 0.02);
        const r3 = normalized(from_plane_angle(bivec.Bivec3f32{ 0, 0, 1 }, ang));
        try expectRotorClose(r3, exp_eps(ln_eps(r3, .{}), .{}), 0.02);
    }
}

test "from_to produces the requested rotation" {
    {
        const r = from_to(@Vector(2, f32){ 1, 0 }, @Vector(2, f32){ 0, 1 });
        try expectRotorClose(@Vector(2, f32){ 0, 1 }, times_vec(r, @Vector(2, f32){ 1, 0 }), 1e-5);
    }
    {
        const r = from_to(@Vector(3, f32){ 1, 0, 0 }, @Vector(3, f32){ 0, 0, 1 });
        try expectRotorClose(@Vector(3, f32){ 0, 0, 1 }, times_vec(r, @Vector(3, f32){ 1, 0, 0 }), 1e-5);
    }
    // Antiparallel (180 degree) case must not NaN.
    {
        const r = from_to(@Vector(3, f32){ 1, 0, 0 }, @Vector(3, f32){ -1, 0, 0 });
        try expectRotorClose(@Vector(3, f32){ -1, 0, 0 }, times_vec(r, @Vector(3, f32){ 1, 0, 0 }), 1e-5);
    }
}

test "apply_spring is exact at dt=0" {
    // 2d
    {
        const start = from_angle(Rotor2f32, std.math.pi);
        var vel: bivec.Bivec2f32 = @splat(0);
        var curr = start;
        apply_spring(&curr, spring_mod.DampedSpringMat(f32).init(.{ .dt = 0, .freq = 10, .ratio = 0.5 }), .{
            .d1 = &vel,
            .target = from_angle(Rotor2f32, std.math.pi / 2.0),
        });
        try std.testing.expectEqual(start, curr);
    }
    // 3d
    {
        const start = from_plane_angle(bivec.Bivec3f32{ 0, 0, 1 }, std.math.pi);
        var vel: bivec.Bivec3f32 = @splat(0);
        var curr = start;
        apply_spring(&curr, spring_mod.DampedSpringMat(f32).init(.{ .dt = 0, .freq = 10, .ratio = 0.5 }), .{
            .d1 = &vel,
            .target = from_plane_angle(bivec.Bivec3f32{ 1, 0, 0 }, std.math.pi / 2.0),
        });
        try std.testing.expectEqual(start, curr);
    }
}

test "apply_spring converges toward target" {
    const target = from_plane_angle(bivec.Bivec3f32{ 0, 0, 1 }, std.math.pi / 2.0);
    var curr = identity(Rotor3f32);
    var vel: bivec.Bivec3f32 = @splat(0);
    for (0..400) |_| {
        apply_spring(&curr, spring_mod.DampedSpringMat(f32).init(.{ .dt = 1.0 / 60.0, .freq = 12, .ratio = 1 }), .{
            .d1 = &vel,
            .target = target,
        });
    }
    try expectRotorClose(target, curr, 0.02);
}

test "apply_spring works on [N]f32 storage" {
    // mr_ecs stores rotors and bivectors as arrays; make sure the seam compiles and runs on them.
    var curr: [4]f32 = identity([4]f32);
    var vel: [3]f32 = .{ 0, 0, 0 };
    const target: [4]f32 = from_plane_angle([3]f32{ 0, 0, 1 }, std.math.pi / 2.0);
    for (0..400) |_| {
        apply_spring(&curr, spring_mod.DampedSpringMat(f32).init(.{ .dt = 1.0 / 60.0, .freq = 12, .ratio = 1 }), .{
            .d1 = &vel,
            .target = target,
        });
    }
    try zml_testing.expect_approx_eq_abs(@as(@Vector(4, f32), target), @as(@Vector(4, f32), curr), 0.02);
}

test "rotor.times disagrees with quat.mul (lane-order landmine)" {
    // Rotor3 and Quat4 share bytes but not lane meaning. Feeding rotors to quat.mul is silently
    // wrong; pin that they genuinely differ so nobody 'optimizes' one into the other.
    const a = normalized(from_plane_angle(bivec.Bivec3f32{ 1, 0, 0 }, 0.9));
    const b = normalized(from_plane_angle(bivec.Bivec3f32{ 0, 1, 0 }, 0.7));
    const via_rotor = times(a, b);
    const qa: quat.Quat4f32 = a;
    const qb: quat.Quat4f32 = b;
    const via_quat: @Vector(4, f32) = quat.mul(qa, qb);
    try std.testing.expect(!std.meta.eql(@as(@Vector(4, f32), via_rotor), via_quat));
}
