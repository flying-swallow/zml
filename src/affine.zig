//! Affine transforms: a `dim x (dim+1)` matrix with the implicit bottom row `[0..0, 1]` elided
//! (2x3 in 2d, 3x4 in 3d), stored column-major as `Mat(T, dim+1, dim)`. Column-major, so the
//! translation is the contiguous last column (`items[dim]`) and the layout uploads directly to a
//! GPU without a transpose. Ported from Games-by-Mason's mr_geom (MIT).
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
    return Mat(meta.Child(Vt), dim + 1, dim);
}

/// The affine matrix type paired with a rotor type `R`.
pub fn AffineForRotor(comptime R: type) type {
    const dim = if (meta.lengthOf(R) == 2) 2 else 3;
    return Mat(meta.Child(R), dim + 1, dim);
}

/// A translation matrix from a translation vector.
pub fn translation(delta: anytype) AffineForVec(@TypeOf(delta)) {
    const M = AffineForVec(@TypeOf(delta));
    const dim = M.rows;
    var result: M = .identity;
    // Translation is the last column, contiguous under column-major storage.
    inline for (0..dim) |i| result.items[dim][i] = delta[i];
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
        inline for (0..dim) |j| result.items[j][i] = rotated[j];
    }
    return result;
}

/// Composes two affine transforms: the result applies `b`, then `a` (`(a*b) * v == a * (b * v)`).
pub fn times(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    const M = @TypeOf(a);
    const dim = M.rows;
    const T = M.Type;
    const Col = @Vector(dim, T);
    var result: M = undefined;
    inline for (0..dim + 1) |j| {
        // Result column j = sum_k (column k of a's linear part) * b[k][j]. For the translation
        // column (j == dim), b's implicit bottom row `[0..0, 1]` adds a's own translation column.
        var acc: Col = if (j == dim) @as(Col, a.items[dim]) else @splat(0);
        inline for (0..dim) |k| {
            acc = @mulAdd(Col, @as(Col, @splat(b.items[j][k])), @as(Col, a.items[k]), acc);
        }
        result.items[j] = acc;
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
    inline for (0..dim) |i| result[i] = m.items[dim][i];
    return result;
}

/// The per-axis scale of the matrix: the magnitude of each transformed basis vector, i.e. the norm
/// of each COLUMN of the linear part. For a `R*S` transform (`applied(scaling, rotation)`) each
/// column of `R*S` is `s_j * (column j of R)`, and `R` is orthonormal, so the column norm is exactly
/// `|s_j|`. (mr_geom's `getScale` used ROW norms, which are only correct for pure scale or a
/// uniformly-scaled rotation; column norms are correct for rotated non-uniform scale too.)
pub fn get_scale(m: anytype) @Vector(@TypeOf(m).rows, @TypeOf(m).Type) {
    const dim = @TypeOf(m).rows;
    const T = @TypeOf(m).Type;
    var result: @Vector(dim, T) = undefined;
    inline for (0..dim) |j| {
        // Column j is contiguous under column-major storage.
        const col: @Vector(dim, T) = m.items[j];
        result[j] = vector.norm(col);
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
        inline for (0..dim) |j| lin[j] = m.items[j][i];
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
        inline for (0..dim) |j| lin[j] = m.items[j][i];
        result[i] = @reduce(.Add, lin * vv) + m.items[dim][i];
    }
    return result;
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

const zml_testing = @import("testing.zig");

test "affine matrix sizes are pinned" {
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(Mat(f32, 3, 2)));
    try std.testing.expectEqual(@as(usize, 48), @sizeOf(Mat(f32, 4, 3)));
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
    // Pure scale: column norms == the scale magnitudes.
    try std.testing.expectEqual(@Vector(3, f32){ 0.5, 2.0, 10.0 }, get_scale(m));
}

test "get_scale uses column norms (correct for rotated non-uniform scale)" {
    // Rotated non-uniform scale: world matrix R*S. The per-axis scale is |s_j| regardless of the
    // rotation, because R is orthonormal so each column of R*S has magnitude |s_j|. Row norms would
    // mix axes here (they'd yield {3, 2, 4} for a 90-degree z rotation of diag(2, 3, 4)).
    const s = @Vector(3, f32){ 2, 3, 4 };
    const m = applied(
        scaling(s),
        rotation(rotor.from_plane_angle(@import("bivec.zig").Bivec3f32{ 0, 0, 1 }, 0.5 * std.math.pi)),
    );
    try zml_testing.expect_approx_eq_abs(s, get_scale(m), 1e-4);
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
    // Mat is always @Vector-free storage ([cols][rows]T), so this mainly guards the vector args.
    const m = translation([3]f32{ 1, 2, 3 });
    try std.testing.expectEqual(@Vector(3, f32){ 1, 2, 3 }, get_translation(m));
}
