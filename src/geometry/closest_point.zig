//! Closest-point-to-origin helpers for a line segment, triangle or tetrahedron, plus the
//! barycentric-coordinate and plane-side tests they build on. Ported from JoltPhysics
//! `Jolt/Geometry/ClosestPoint.h`.
//!
//! Every routine works relative to the origin: pass the feature's points already translated by the
//! query point (subtract `p` from each vertex), or use the `*_to(p, ...)` convenience wrappers. The
//! returned `set` is a feature bitmask matching Jolt's convention — for a triangle
//! `1 = a, 2 = b, 4 = c`, so an edge has two bits set and the interior has three; for a tetrahedron
//! `1 = a, 2 = b, 4 = c, 8 = d`. These bitmasks are what a GJK simplex uses to drop unused vertices.

const std = @import("std");
const zml = @import("../root.zig");

/// Result of a closest-point query: the closest point and the feature bitmask that was closest.
pub fn ClosestPointResult(comptime T: type) type {
    return struct { point: @Vector(3, T), set: u32 };
}

/// Barycentric coordinates on a line. `u`/`v` are always set (fallback to the nearer endpoint when
/// `ok` is false, i.e. the two points coincide).
pub fn BaryLine(comptime T: type) type {
    return struct { u: T, v: T, ok: bool };
}

/// Barycentric coordinates on a triangle. `u`/`v`/`w` are always set (fallback along the longest
/// edge when `ok` is false, i.e. the triangle is degenerate).
pub fn BaryTriangle(comptime T: type) type {
    return struct { u: T, v: T, w: T, ok: bool };
}

/// `a*b - c*d` computed with fused multiply-adds to cancel the rounding of the two products
/// (Kahan's difference of products, as in Jolt's `DifferenceOfProducts`).
inline fn difference_of_products(comptime T: type, a: T, b: T, c: T, d: T) T {
    const cd = c * d;
    const err = @mulAdd(T, -c, d, cd); // rounding error of `cd`
    const dop = @mulAdd(T, a, b, -cd);
    return dop + err;
}

/// Cross product using `difference_of_products` for each component (Jolt's `CrossPrecise`), used for
/// the plane normals where cancellation matters most.
inline fn cross_precise(comptime T: type, a: @Vector(3, T), b: @Vector(3, T)) @Vector(3, T) {
    return .{
        difference_of_products(T, a[1], b[2], a[2], b[1]),
        difference_of_products(T, a[2], b[0], a[0], b[2]),
        difference_of_products(T, a[0], b[1], a[1], b[0]),
    };
}

inline fn splat(comptime T: type, s: T) @Vector(3, T) {
    return @splat(s);
}

/// Barycentric coordinates of the closest point on infinite line (a, b) to the origin. The point is
/// `a*u + b*v`. `ok` is false (with a nearest-endpoint fallback) when a and b coincide.
pub fn barycentric_line(comptime T: type, a: @Vector(3, T), b: @Vector(3, T)) BaryLine(T) {
    const ab = b - a;
    const denom = zml.vec.norm_sqr(ab);
    if (denom < std.math.floatEps(T) * std.math.floatEps(T)) {
        // Degenerate segment: pick the endpoint closer to the origin.
        if (zml.vec.norm_sqr(a) < zml.vec.norm_sqr(b)) {
            return .{ .u = 1, .v = 0, .ok = false };
        } else {
            return .{ .u = 0, .v = 1, .ok = false };
        }
    }
    const v = -zml.vec.dot(a, ab) / denom;
    return .{ .u = 1 - v, .v = v, .ok = true };
}

/// Barycentric coordinates of the closest point on the plane of triangle (a, b, c) to the origin.
/// The point is `a*u + b*v + c*w`. Always includes the shortest edge in the computation for accuracy
/// (Ericson RTCD). `ok` is false (with a longest-edge fallback) when the triangle is degenerate.
pub fn barycentric_triangle(comptime T: type, a: @Vector(3, T), b: @Vector(3, T), c: @Vector(3, T)) BaryTriangle(T) {
    const v0 = b - a;
    const v1 = c - a;
    const v2 = c - b;
    const d00 = zml.vec.norm_sqr(v0);
    const d11 = zml.vec.norm_sqr(v1);
    const d22 = zml.vec.norm_sqr(v2);
    if (d00 <= d22) {
        const d01 = zml.vec.dot(v0, v1);
        const denom = difference_of_products(T, d00, d11, d01, d01);
        if (denom < 1.0e-12) {
            // Degenerate: coordinates along the longest edge.
            if (d00 > d11) {
                const bl = barycentric_line(T, a, b);
                return .{ .u = bl.u, .v = bl.v, .w = 0, .ok = false };
            } else {
                const bl = barycentric_line(T, a, c);
                return .{ .u = bl.u, .v = 0, .w = bl.v, .ok = false };
            }
        }
        const a0 = zml.vec.dot(a, v0);
        const a1 = zml.vec.dot(a, v1);
        const v = difference_of_products(T, d01, a1, d11, a0) / denom;
        const w = difference_of_products(T, d01, a0, d00, a1) / denom;
        return .{ .u = 1 - v - w, .v = v, .w = w, .ok = true };
    } else {
        const d12 = zml.vec.dot(v1, v2);
        const denom = difference_of_products(T, d11, d22, d12, d12);
        if (denom < 1.0e-12) {
            if (d11 > d22) {
                const bl = barycentric_line(T, a, c);
                return .{ .u = bl.u, .v = 0, .w = bl.v, .ok = false };
            } else {
                const bl = barycentric_line(T, b, c);
                return .{ .u = 0, .v = bl.u, .w = bl.v, .ok = false };
            }
        }
        const c1 = zml.vec.dot(c, v1);
        const c2 = zml.vec.dot(c, v2);
        const u = difference_of_products(T, d22, c1, d12, c2) / denom;
        const v = difference_of_products(T, d11, c2, d12, c1) / denom;
        return .{ .u = u, .v = v, .w = 1 - u - v, .ok = true };
    }
}

/// Closest point on line segment (a, b) to the origin. `set`: 1 = a, 2 = b, 3 = interior of ab.
pub fn closest_point_on_line(comptime T: type, a: @Vector(3, T), b: @Vector(3, T)) ClosestPointResult(T) {
    const bl = barycentric_line(T, a, b);
    if (bl.v <= 0) return .{ .point = a, .set = 0b0001 };
    if (bl.u <= 0) return .{ .point = b, .set = 0b0010 };
    return .{ .point = a * splat(T, bl.u) + b * splat(T, bl.v), .set = 0b0011 };
}

/// Closest point on triangle (a, b, c) to the origin. `set` is a feature bitmask (1 = a, 2 = b,
/// 4 = c; edges have two bits, the interior 0b0111). When `must_include_c` is true the function
/// assumes the closest feature includes c and does less work; if that assumption is wrong it still
/// returns a valid closest point to another feature. Ericson RTCD.
pub fn closest_point_on_triangle(comptime T: type, comptime must_include_c: bool, in_a: @Vector(3, T), in_b: @Vector(3, T), in_c: @Vector(3, T)) ClosestPointResult(T) {
    const V = @Vector(3, T);
    const eps = std.math.floatEps(T);
    const eps_sq = eps * eps;

    // Include the shortest of edges ba/bc so the normal uses the two shortest edges (more accurate):
    // ensure ab is shorter than bc by swapping a and c when it is not.
    const ba = in_a - in_b;
    const bc = in_c - in_b;
    const swap = zml.vec.norm_sqr(bc) < zml.vec.norm_sqr(ba);
    const a = if (swap) in_c else in_a;
    const c = if (swap) in_a else in_c;

    const ab = in_b - a;
    const ac = c - a;
    const n = cross_precise(T, ab, ac);
    const n_len_sq = zml.vec.norm_sqr(n);

    if (n_len_sq < 1.0e-10) {
        // Degenerate triangle: fall back to vertices and edges. Prefer vertices over edges (fewer
        // bits set), so test them first.
        var closest_set: u32 = 0b0100;
        var closest_point: V = in_c;
        var best: T = zml.vec.norm_sqr(in_c);

        if (!must_include_c) {
            const a_sq = zml.vec.norm_sqr(in_a);
            if (a_sq < best) {
                closest_set = 0b0001;
                closest_point = in_a;
                best = a_sq;
            }
            const b_sq = zml.vec.norm_sqr(in_b);
            if (b_sq < best) {
                closest_set = 0b0010;
                closest_point = in_b;
                best = b_sq;
            }
        }
        // Edge AC
        const ac_len_sq = zml.vec.norm_sqr(ac);
        if (ac_len_sq > eps_sq) {
            const t = @min(@max(-zml.vec.dot(a, ac) / ac_len_sq, 0), 1);
            const q = a + ac * splat(T, t);
            const q_sq = zml.vec.norm_sqr(q);
            if (q_sq < best) {
                closest_set = 0b0101;
                closest_point = q;
                best = q_sq;
            }
        }
        // Edge BC
        const bc2 = in_c - in_b;
        const bc_len_sq = zml.vec.norm_sqr(bc2);
        if (bc_len_sq > eps_sq) {
            const t = @min(@max(-zml.vec.dot(in_b, bc2) / bc_len_sq, 0), 1);
            const q = in_b + bc2 * splat(T, t);
            const q_sq = zml.vec.norm_sqr(q);
            if (q_sq < best) {
                closest_set = 0b0110;
                closest_point = q;
                best = q_sq;
            }
        }
        if (!must_include_c) {
            // Edge AB
            const ab2 = in_b - in_a;
            const ab_len_sq = zml.vec.norm_sqr(ab2);
            if (ab_len_sq > eps_sq) {
                const t = @min(@max(-zml.vec.dot(in_a, ab2) / ab_len_sq, 0), 1);
                const q = in_a + ab2 * splat(T, t);
                const q_sq = zml.vec.norm_sqr(q);
                if (q_sq < best) {
                    closest_set = 0b0011;
                    closest_point = q;
                    best = q_sq;
                }
            }
        }
        return .{ .point = closest_point, .set = closest_set };
    }

    // Non-degenerate: Voronoi region tests (origin = p = 0).
    const ap = -a;
    const d1 = zml.vec.dot(ab, ap);
    const d2 = zml.vec.dot(ac, ap);
    if (d1 <= 0 and d2 <= 0) return .{ .point = a, .set = if (swap) 0b0100 else 0b0001 }; // vertex A

    const bp = -in_b;
    const d3 = zml.vec.dot(ab, bp);
    const d4 = zml.vec.dot(ac, bp);
    if (d3 >= 0 and d4 <= d3) return .{ .point = in_b, .set = 0b0010 }; // vertex B

    if (d1 * d4 <= d3 * d2 and d1 >= 0 and d3 <= 0) {
        const v = d1 / (d1 - d3);
        return .{ .point = a + ab * splat(T, v), .set = if (swap) 0b0110 else 0b0011 }; // edge AB
    }

    const cp = -c;
    const d5 = zml.vec.dot(ab, cp);
    const d6 = zml.vec.dot(ac, cp);
    if (d6 >= 0 and d5 <= d6) return .{ .point = c, .set = if (swap) 0b0001 else 0b0100 }; // vertex C

    if (d5 * d2 <= d1 * d6 and d2 >= 0 and d6 <= 0) {
        const w = d2 / (d2 - d6);
        return .{ .point = a + ac * splat(T, w), .set = 0b0101 }; // edge AC
    }

    const d4_d3 = d4 - d3;
    const d5_d6 = d5 - d6;
    if (d3 * d6 <= d5 * d4 and d4_d3 >= 0 and d5_d6 >= 0) {
        const w = d4_d3 / (d4_d3 + d5_d6);
        return .{ .point = in_b + (c - in_b) * splat(T, w), .set = if (swap) 0b0011 else 0b0110 }; // edge BC
    }

    // Interior: project the origin onto the plane. More accurate than reconstructing from
    // barycentric coordinates.
    const scale = zml.vec.dot(a + in_b + c, n) / (3 * n_len_sq);
    return .{ .point = n * splat(T, scale), .set = 0b0111 };
}

/// True if the origin is on the far side of the plane through (a, b, c) from d (d marks the front).
pub fn origin_outside_of_plane(comptime T: type, a: @Vector(3, T), b: @Vector(3, T), c: @Vector(3, T), d: @Vector(3, T)) bool {
    const n = cross_precise(T, b - a, c - a);
    const signp = zml.vec.dot(a, n); // [AP AB AC], with P = origin
    const signd = zml.vec.dot(d - a, n); // [AD AB AC]
    // Opposite sides when the product is positive (the minus sign of signp is folded in).
    return signp * signd > -std.math.floatEps(T);
}

/// For each face of tetrahedron (a, b, c, d), whether the origin is outside it. Components are the
/// faces ABC, ACD, ADB, BDC in order. A degenerate tetrahedron reports outside for every face.
pub fn origin_outside_of_tetrahedron_planes(comptime T: type, a: @Vector(3, T), b: @Vector(3, T), c: @Vector(3, T), d: @Vector(3, T)) @Vector(4, bool) {
    const ab = b - a;
    const ac = c - a;
    const ad = d - a;
    const bd = d - b;
    const bc = c - b;

    const ab_cross_ac = cross_precise(T, ab, ac);
    const ac_cross_ad = cross_precise(T, ac, ad);
    const ad_cross_ab = cross_precise(T, ad, ab);
    const bd_cross_bc = cross_precise(T, bd, bc);

    // For each plane, which side the origin is on.
    const signp = @Vector(4, T){
        zml.vec.dot(a, ab_cross_ac), // ABC
        zml.vec.dot(a, ac_cross_ad), // ACD
        zml.vec.dot(a, ad_cross_ab), // ADB
        zml.vec.dot(b, bd_cross_bc), // BDC
    };
    // For each plane, which side is outside (determined by the 4th point).
    const signd = @Vector(4, T){
        zml.vec.dot(ad, ab_cross_ac), // D
        zml.vec.dot(ab, ac_cross_ad), // B
        zml.vec.dot(ac, ad_cross_ab), // C
        -zml.vec.dot(ab, bd_cross_bc), // A
    };

    const zero: @Vector(4, T) = @splat(0);
    const eps: T = std.math.floatEps(T);
    const neg = signd < zero;
    if (!@reduce(.Or, neg)) {
        // All windings positive: origin outside where signp >= -eps.
        return signp >= @as(@Vector(4, T), @splat(-eps));
    }
    if (@reduce(.And, neg)) {
        // All windings negative: origin outside where signp <= eps.
        return signp <= @as(@Vector(4, T), @splat(eps));
    }
    // Mixed signs: degenerate tetrahedron, treat the origin as outside every face.
    return @splat(true);
}

/// Closest point on tetrahedron (a, b, c, d) to the origin. `set` is a feature bitmask (1 = a,
/// 2 = b, 4 = c, 8 = d); an edge has two bits, a face three, the interior four. When
/// `must_include_d` is true the function assumes the closest feature includes d and does less work.
pub fn closest_point_on_tetrahedron(comptime T: type, comptime must_include_d: bool, a: @Vector(3, T), b: @Vector(3, T), c: @Vector(3, T), d: @Vector(3, T)) ClosestPointResult(T) {
    const V = @Vector(3, T);
    // Assume the origin is inside all half-spaces (closest to itself) until a face says otherwise.
    var closest_set: u32 = 0b1111;
    var closest_point: V = @splat(0);
    var best: T = std.math.floatMax(T);

    const oop = origin_outside_of_tetrahedron_planes(T, a, b, c, d);

    if (oop[0]) { // face ABC
        if (must_include_d) {
            // ABC cannot contain d, and the origin cannot be interior, so a is the closest we keep.
            closest_set = 0b0001;
            closest_point = a;
        } else {
            const r = closest_point_on_triangle(T, false, a, b, c);
            closest_point = r.point;
            closest_set = r.set;
        }
        best = zml.vec.norm_sqr(closest_point);
    }
    if (oop[1]) { // face ACD
        const r = closest_point_on_triangle(T, must_include_d, a, c, d);
        const q_sq = zml.vec.norm_sqr(r.point);
        if (q_sq < best) {
            best = q_sq;
            closest_point = r.point;
            closest_set = (r.set & 0b0001) + ((r.set & 0b0110) << 1);
        }
    }
    if (oop[2]) { // face ADB (kept as triangle A, B, D for GJK consistency)
        const r = closest_point_on_triangle(T, must_include_d, a, b, d);
        const q_sq = zml.vec.norm_sqr(r.point);
        if (q_sq < best) {
            best = q_sq;
            closest_point = r.point;
            closest_set = (r.set & 0b0011) + ((r.set & 0b0100) << 1);
        }
    }
    if (oop[3]) { // face BDC (kept as triangle B, C, D for GJK consistency)
        const r = closest_point_on_triangle(T, must_include_d, b, c, d);
        const q_sq = zml.vec.norm_sqr(r.point);
        if (q_sq < best) {
            closest_point = r.point;
            closest_set = r.set << 1;
        }
    }
    return .{ .point = closest_point, .set = closest_set };
}

// ---- Convenience wrappers taking an explicit query point `p` ----

/// Closest point on segment (a, b) to `p`.
pub fn closest_point_on_line_to(comptime T: type, p: @Vector(3, T), a: @Vector(3, T), b: @Vector(3, T)) @Vector(3, T) {
    return closest_point_on_line(T, a - p, b - p).point + p;
}

/// Closest point on triangle (a, b, c) to `p`.
pub fn closest_point_on_triangle_to(comptime T: type, p: @Vector(3, T), a: @Vector(3, T), b: @Vector(3, T), c: @Vector(3, T)) @Vector(3, T) {
    return closest_point_on_triangle(T, false, a - p, b - p, c - p).point + p;
}

/// Closest point on tetrahedron (a, b, c, d) to `p`.
pub fn closest_point_on_tetrahedron_to(comptime T: type, p: @Vector(3, T), a: @Vector(3, T), b: @Vector(3, T), c: @Vector(3, T), d: @Vector(3, T)) @Vector(3, T) {
    return closest_point_on_tetrahedron(T, false, a - p, b - p, c - p, d - p).point + p;
}

test barycentric_line {
    // Origin projects to the midpoint of a segment centred on it.
    const bl = barycentric_line(f32, .{ -1, 0, 0 }, .{ 1, 0, 0 });
    try std.testing.expect(bl.ok);
    try std.testing.expectApproxEqRel(@as(f32, 0.5), bl.u, 1.0e-6);
    try std.testing.expectApproxEqRel(@as(f32, 0.5), bl.v, 1.0e-6);
    // Degenerate segment reports the nearer endpoint.
    const dg = barycentric_line(f32, .{ 2, 0, 0 }, .{ 2, 0, 0 });
    try std.testing.expect(!dg.ok);
}

test closest_point_on_line {
    // Origin beyond endpoint a -> a is closest.
    const r1 = closest_point_on_line(f32, .{ 1, 0, 0 }, .{ 2, 0, 0 });
    try std.testing.expect(zml.vec.is_close_default(r1.point, .{ 1, 0, 0 }));
    try std.testing.expectEqual(@as(u32, 0b0001), r1.set);
    // Origin projects onto the interior.
    const r2 = closest_point_on_line(f32, .{ -1, 1, 0 }, .{ 1, 1, 0 });
    try std.testing.expect(zml.vec.is_close_default(r2.point, .{ 0, 1, 0 }));
    try std.testing.expectEqual(@as(u32, 0b0011), r2.set);
}

test closest_point_on_triangle {
    // Triangle in the z = 1 plane straddling the origin's projection: closest point is (0,0,1),
    // interior feature.
    const a: @Vector(3, f32) = .{ -1, -1, 1 };
    const b: @Vector(3, f32) = .{ 3, -1, 1 };
    const c: @Vector(3, f32) = .{ -1, 3, 1 };
    const r = closest_point_on_triangle(f32, false, a, b, c);
    try std.testing.expect(zml.vec.is_close_default(r.point, .{ 0, 0, 1 }));
    try std.testing.expectEqual(@as(u32, 0b0111), r.set);

    // Origin nearest a vertex: shift the triangle far into +x so the a-corner is closest.
    const r2 = closest_point_on_triangle(f32, false, .{ 2, 0, 0 }, .{ 4, 1, 0 }, .{ 4, -1, 0 });
    try std.testing.expect(zml.vec.is_close_default(r2.point, .{ 2, 0, 0 }));
    try std.testing.expectEqual(@as(u32, 0b0001), r2.set);
}

test "closest_point_on_triangle to query point" {
    const p: @Vector(3, f32) = .{ 0, 0, 5 };
    const q = closest_point_on_triangle_to(f32, p, .{ -1, -1, 0 }, .{ 3, -1, 0 }, .{ -1, 3, 0 });
    try std.testing.expect(zml.vec.is_close_default(q, .{ 0, 0, 0 }));
}

test closest_point_on_tetrahedron {
    // Tetrahedron with the origin strictly inside -> interior feature, closest point is the origin.
    const a: @Vector(3, f32) = .{ 1, 1, 1 };
    const b: @Vector(3, f32) = .{ 1, -1, -1 };
    const c: @Vector(3, f32) = .{ -1, 1, -1 };
    const d: @Vector(3, f32) = .{ -1, -1, 1 };
    const r = closest_point_on_tetrahedron(f32, false, a, b, c, d);
    try std.testing.expect(zml.vec.is_close_default(r.point, .{ 0, 0, 0 }));
    try std.testing.expectEqual(@as(u32, 0b1111), r.set);

    // Move the whole tetrahedron +2 in x. Vertices c and d sit at x = 1 while a and b move to x = 3,
    // so the two faces meeting at edge cd slope away and the nearest feature is edge cd, closest
    // point (1, 0, 0).
    const off: @Vector(3, f32) = .{ 2, 0, 0 };
    const r2 = closest_point_on_tetrahedron(f32, false, a + off, b + off, c + off, d + off);
    try std.testing.expect(zml.vec.is_close_default(r2.point, .{ 1, 0, 0 }));
    try std.testing.expectEqual(@as(u32, 0b1100), r2.set); // edge cd (c | d)
}

test origin_outside_of_plane {
    // Plane z = 1 with front side toward d at z = 2; origin at z = 0 is on the far side -> outside.
    const a: @Vector(3, f32) = .{ 0, 0, 1 };
    const b: @Vector(3, f32) = .{ 1, 0, 1 };
    const c: @Vector(3, f32) = .{ 0, 1, 1 };
    const d: @Vector(3, f32) = .{ 0, 0, 2 };
    try std.testing.expect(origin_outside_of_plane(f32, a, b, c, d));
    // With the front side toward the origin (d at z = 0.5), the origin is not outside.
    try std.testing.expect(!origin_outside_of_plane(f32, a, b, c, .{ 0, 0, 0.5 }));
}
