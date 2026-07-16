//! Affine transforms stored as a `Mat(T, dim, dim+1)` with the implicit bottom row `[0..0, 1]`
//! elided: 2x3 in 2d, 3x4 in 3d. Row-major, so `items[i][dim]` is the translation on axis i and the
//! layout is byte-compatible with a compact GPU transform. Ported from Games-by-Mason's mr_geom
//! (MIT).
//!
//! These are FREE FUNCTIONS, not `Mat` methods, on purpose: a `Mat(T, dim, dim+1)` is also a valid
//! general (dim)x(dim+1) matrix, and `affine.times` (implicit `[0..0,1]` row) means something
//! different from `Mat.mul`. Keeping them separate stops the two multiplications from being
//! confused, and sidesteps the name clashes with `Mat.scale` (a modifier) and `Mat.get_scale`
//! (column norms, asserts rows >= 3).

const std = @import("std");
const Mat = @import("matrix.zig").Mat;
const rotor = @import("rotor.zig");
const meta = @import("meta.zig");
const vector = @import("vector.zig");

/// The affine matrix type acting on a length-N vector type `Vt`.
pub fn AffineForVec(comptime Vt: type) type {
    const dim = meta.lengthOf(Vt);
    return Mat(meta.Child(Vt), dim, dim + 1);
}

/// The affine matrix type paired with a rotor type `R`.
pub fn AffineForRotor(comptime R: type) type {
    const dim = if (meta.lengthOf(R) == 2) 2 else 3;
    return Mat(meta.Child(R), dim, dim + 1);
}

/// A translation matrix from a translation vector.
pub fn translation(delta: anytype) AffineForVec(@TypeOf(delta)) {
    const M = AffineForVec(@TypeOf(delta));
    const dim = M.rows;
    var result: M = .identity;
    inline for (0..dim) |i| result.items[i][dim] = delta[i];
    return result;
}

/// A scale matrix from a per-axis scale vector. Named `scaling` (not `scale`) so it never collides
/// with `Mat.scale`, which is a modifier rather than a constructor.
pub fn scaling(factors: anytype) AffineForVec(@TypeOf(factors)) {
    const M = AffineForVec(@TypeOf(factors));
    const dim = M.rows;
    var result: M = .zero;
    inline for (0..dim) |i| result.items[i][i] = factors[i];
    return result;
}

/// A rotation matrix from a rotor. `M * v == rotor.times_vec(v)`.
pub fn rotation(r: anytype) AffineForRotor(@TypeOf(r)) {
    const M = AffineForRotor(@TypeOf(r));
    const dim = M.rows;
    const T = M.Type;
    // Row i's linear part is the inverse rotor applied to axis i, matching mr_geom exactly (this is
    // R itself, since applying the inverse rotor to e_i yields row i of R).
    const inv = rotor.inverse(r);
    var result: M = .zero;
    inline for (0..dim) |i| {
        var axis: @Vector(dim, T) = @splat(0);
        axis[i] = 1;
        const rotated: @Vector(dim, T) = rotor.times_vec(inv, axis);
        inline for (0..dim) |j| result.items[i][j] = rotated[j];
    }
    return result;
}

/// Composes two affine transforms: the result applies `b`, then `a` (`(a*b) * v == a * (b * v)`).
pub fn times(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    const M = @TypeOf(a);
    const dim = M.rows;
    const T = M.Type;
    const Row = @Vector(dim + 1, T);
    var result: M = undefined;
    inline for (0..dim) |i| {
        // b's implicit bottom row is [0..0, 1]; a's row i contributes a.items[i][dim] to it.
        var acc: Row = @splat(0);
        acc[dim] = a.items[i][dim];
        inline for (0..dim) |k| {
            const bk: Row = b.items[k];
            acc = @mulAdd(Row, @as(Row, @splat(a.items[i][k])), bk, acc);
        }
        result.items[i] = acc;
    }
    return result;
}

/// The same as `times` with the arguments reversed -- often more intuitive: `a.applied(b)` reads as
/// "a, with b applied after". Equal to `times(b, a)`.
pub fn applied(self: anytype, other: @TypeOf(self)) @TypeOf(self) {
    return times(other, self);
}

/// The translation component of the matrix.
pub fn get_translation(m: anytype) @Vector(@TypeOf(m).rows, @TypeOf(m).Type) {
    const dim = @TypeOf(m).rows;
    var result: @Vector(dim, @TypeOf(m).Type) = undefined;
    inline for (0..dim) |i| result[i] = m.items[i][dim];
    return result;
}

/// The per-axis scale of the matrix.
///
/// ⚠️ Ported bug: this uses ROW norms, but the true per-axis scale is COLUMN norms. They agree for
/// pure scale and for uniformly-scaled rotations, but disagree for a rotated non-uniform scale (the
/// exact case `Transform.sync` builds). This matches mr_geom's `getScale` verbatim so the migration
/// changes no behavior; the fix (column norms) is a deliberately separate follow-up.
pub fn get_scale(m: anytype) @Vector(@TypeOf(m).rows, @TypeOf(m).Type) {
    const dim = @TypeOf(m).rows;
    const T = @TypeOf(m).Type;
    var result: @Vector(dim, T) = undefined;
    inline for (0..dim) |i| {
        var lin: @Vector(dim, T) = undefined;
        inline for (0..dim) |j| lin[j] = m.items[i][j];
        result[i] = vector.norm(lin);
    }
    return result;
}

/// Transforms a direction by the matrix (ignores translation).
pub fn times_dir(m: anytype, v: anytype) @Vector(@TypeOf(m).rows, @TypeOf(m).Type) {
    const dim = @TypeOf(m).rows;
    const T = @TypeOf(m).Type;
    const vv: @Vector(dim, T) = v;
    var result: @Vector(dim, T) = undefined;
    inline for (0..dim) |i| {
        var lin: @Vector(dim, T) = undefined;
        inline for (0..dim) |j| lin[j] = m.items[i][j];
        result[i] = @reduce(.Add, lin * vv);
    }
    return result;
}

/// Transforms a point by the matrix (applies translation).
pub fn times_point(m: anytype, v: anytype) @Vector(@TypeOf(m).rows, @TypeOf(m).Type) {
    const dim = @TypeOf(m).rows;
    const T = @TypeOf(m).Type;
    const vv: @Vector(dim, T) = v;
    var result: @Vector(dim, T) = undefined;
    inline for (0..dim) |i| {
        var lin: @Vector(dim, T) = undefined;
        inline for (0..dim) |j| lin[j] = m.items[i][j];
        result[i] = @reduce(.Add, lin * vv) + m.items[i][dim];
    }
    return result;
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

const zml_testing = @import("testing.zig");

test "affine matrix sizes are pinned" {
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(Mat(f32, 2, 3)));
    try std.testing.expectEqual(@as(usize, 48), @sizeOf(Mat(f32, 3, 4)));
}

test "translation" {
    const m = translation(@Vector(3, f32){ 1, 2, 3 });
    try std.testing.expectEqual(@Vector(3, f32){ 1, 2, 3 }, get_translation(m));
    try std.testing.expectEqual(@Vector(3, f32){ 3, 5, 7 }, times_point(m, @Vector(3, f32){ 2, 3, 4 }));
    // Directions ignore translation.
    try std.testing.expectEqual(@Vector(3, f32){ 2, 3, 4 }, times_dir(m, @Vector(3, f32){ 2, 3, 4 }));
}

test "scaling" {
    const m = scaling(@Vector(3, f32){ 0.5, -2.0, 10.0 });
    try std.testing.expectEqual(@Vector(3, f32){ 0.5, -6.0, 50.0 }, times_point(m, @Vector(3, f32){ 1, 3, 5 }));
    try std.testing.expectEqual(@Vector(3, f32){ 0, 0, 0 }, get_translation(m));
    // Pure scale: row norms == the scale magnitudes.
    try std.testing.expectEqual(@Vector(3, f32){ 0.5, 2.0, 10.0 }, get_scale(m));
}

test "rotation applies the rotor" {
    const r = rotor.from_to(@Vector(3, f32){ 1, 0, 0 }, @Vector(3, f32){ 0, 1, 0 });
    const m = rotation(r);
    try zml_testing.expect_approx_eq_abs(@Vector(3, f32){ 0, 1, 0 }, times_point(m, @Vector(3, f32){ 1, 0, 0 }), 1e-5);
    try std.testing.expectEqual(@Vector(3, f32){ 0, 0, 0 }, get_translation(m));
}

test "times / applied compose T*R*S like sync" {
    const t = translation(@Vector(3, f32){ 1, 2, 3 });
    const r = rotation(rotor.from_to(@Vector(3, f32){ 0, 1, 0 }, @Vector(3, f32){ 1, 0, 0 })); // y -> x
    const s = scaling(@Vector(3, f32){ 0.5, 3.0, 4.0 });

    // sync builds scale.applied(rotation).applied(translation), i.e. the world matrix T*R*S.
    const via_applied = applied(applied(s, r), t);
    const via_times = times(times(t, r), s);
    try zml_testing.expect_approx_eq_abs(via_applied, via_times, 1e-5);

    // T*R*S * origin = translation.
    try zml_testing.expect_approx_eq_abs(@Vector(3, f32){ 1, 2, 3 }, times_point(via_times, @Vector(3, f32){ 0, 0, 0 }), 1e-4);
    // model +y -> scaled to (0,3,0) -> rotated y->x to (3,0,0) -> translated to (4,2,3).
    try zml_testing.expect_approx_eq_abs(@Vector(3, f32){ 4, 2, 3 }, times_point(via_times, @Vector(3, f32){ 0, 1, 0 }), 1e-4);
}

test "get_scale is invariant to uniform-scaled rotation (row==column norms)" {
    const s = @Vector(3, f32){ 2, 2, 2 };
    const m = applied(
        scaling(s),
        rotation(rotor.from_plane_angle(@import("bivec.zig").Bivec3f32{ 0, 0, 1 }, 0.5 * std.math.pi)),
    );
    try zml_testing.expect_approx_eq_abs(s, get_scale(m), 1e-4);
}

test "2d affine round trip" {
    const t = translation(@Vector(2, f32){ 3, 4 });
    const r = rotation(rotor.from_to(@Vector(2, f32){ 1, 0 }, @Vector(2, f32){ 0, 1 })); // x -> y
    const m = times(t, r); // translation after rotation
    // Rotate +x to +y, then translate by (3,4): point (1,0) -> (0,1) -> (3,5).
    try zml_testing.expect_approx_eq_abs(@Vector(2, f32){ 3, 5 }, times_point(m, @Vector(2, f32){ 1, 0 }), 1e-5);
}

test "affine ops work on [N]f32-backed matrices" {
    // Mat is always @Vector-free storage ([rows][cols]T), so this mainly guards the vector args.
    const m = translation([3]f32{ 1, 2, 3 });
    try std.testing.expectEqual(@Vector(3, f32){ 1, 2, 3 }, get_translation(m));
}
