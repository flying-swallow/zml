const std = @import("std");
const zml = @import("../root.zig");
const Mat = @import("../matrix.zig").Mat;
const geometry = @import("../geometry.zig");
const Sphere = @import("sphere.zig").Sphere;
const vector = @import("../vector.zig");

// aabb
pub fn overlap_sphere_sphere(a: anytype, b: anytype) bool {
    comptime {
        std.debug.assert(@TypeOf(a).child == @TypeOf(b).child); 
        std.debug.assert(@TypeOf(a).primative_type == .Sphere);
        std.debug.assert(@TypeOf(b).primative_type == .Sphere);
    }
    return vector.norm_sqr(a.center - b.center) <= (a.radius + b.radius) * (a.radius + b.radius);
}

pub fn overlap_aabb_aabb(a: anytype, b: anytype) bool {
    comptime {
        std.debug.assert(@TypeOf(a).child == @TypeOf(b).child);
        std.debug.assert(@TypeOf(a).primative_type == .AABB);
        std.debug.assert(@TypeOf(b).primative_type == .AABB);
    }
    return !@reduce(.Or, (a.min > b.max) | (a.max < b.min));
}

pub fn overlap_aabb_plane(a: anytype, b: anytype) bool {
    comptime {
        std.debug.assert(@TypeOf(a).child == @TypeOf(b).child);
        std.debug.assert(@TypeOf(a).primative_type == .AABB);
        std.debug.assert(@TypeOf(b).primative_type == .Plane);
    }
    const dist_normal = b.signed_distance(a.get_support(b.normal));
    const dist_min_normal = b.signed_distance(a.get_support(-b.normal));
    return dist_normal * dist_min_normal <= 0;
}

pub fn overlap_aabb_sphere(a: anytype, b: anytype) bool {
    comptime {
        std.debug.assert(@TypeOf(a).child == @TypeOf(b).child);
        std.debug.assert(@TypeOf(a).primative_type == .AABB);
        std.debug.assert(@TypeOf(b).primative_type == .Sphere);
    }
    return a.get_sqr_distance_to(b.center) <= b.radius * b.radius;
}

pub fn overlap_aabb_aabb_4(a: anytype, minX: @Vector(4, @TypeOf(a).child), maxX: @Vector(4, @TypeOf(a).child), minY: @Vector(4, @TypeOf(a).child), maxY: @Vector(4, @TypeOf(a).child), minZ: @Vector(4, @TypeOf(a).child), maxZ: @Vector(4, @TypeOf(a).child)) @Vector(4, bool) {
    comptime {
        std.debug.assert(@TypeOf(a).primative_type == .AABB);
    }
    const box1_minx = @as(@Vector(4, @TypeOf(a).child), @splat(a.min[0]));
    const box1_miny = @as(@Vector(4, @TypeOf(a).child), @splat(a.min[1]));
    const box1_minz = @as(@Vector(4, @TypeOf(a).child), @splat(a.min[2]));
    const box1_maxx = @as(@Vector(4, @TypeOf(a).child), @splat(a.max[0]));
    const box1_maxy = @as(@Vector(4, @TypeOf(a).child), @splat(a.max[1]));
    const box1_maxz = @as(@Vector(4, @TypeOf(a).child), @splat(a.max[2]));

    const nooverlap_x = (box1_minx > maxX) | (box1_maxx < minX);
    const nooverlap_y = (box1_miny > maxY) | (box1_maxy < minY);
    const nooverlap_z = (box1_minz > maxZ) | (box1_maxz < minZ);
    return !(nooverlap_x | nooverlap_y | nooverlap_z);
}

/// Axis-aligned box vs triangle overlap using the separating-axis theorem
/// (Akenine-Möller). The triangle is given by its three vertices.
pub fn overlap_aabb_triangle(
    aabb: anytype,
    t0: @Vector(3, @TypeOf(aabb).child),
    t1: @Vector(3, @TypeOf(aabb).child),
    t2: @Vector(3, @TypeOf(aabb).child),
) bool {
    comptime std.debug.assert(@TypeOf(aabb).primative_type == .AABB);
    const T = @TypeOf(aabb).child;
    const V = @Vector(3, T);

    const c = aabb.get_center();
    const h = aabb.get_size() * @as(V, @splat(0.5));
    const v0 = t0 - c;
    const v1 = t1 - c;
    const v2 = t2 - c;
    const edges = [3]V{ v1 - v0, v2 - v1, v0 - v2 };
    const verts = [3]V{ v0, v1, v2 };

    // 9 axes: cross products of the triangle edges with the box axes.
    inline for (0..3) |ei| {
        inline for (0..3) |ai| {
            var unit: V = .{ 0, 0, 0 };
            unit[ai] = 1;
            const axis = vector.cross(edges[ei], unit);
            var pmin: T = std.math.inf(T);
            var pmax: T = -std.math.inf(T);
            inline for (0..3) |k| {
                const p = vector.dot(axis, verts[k]);
                pmin = @min(pmin, p);
                pmax = @max(pmax, p);
            }
            const r = vector.dot(h, @abs(axis));
            if (pmin > r or pmax < -r) return false;
        }
    }

    // 3 box face normals: the triangle's AABB must overlap the box.
    inline for (0..3) |i| {
        const tmin = @min(v0[i], @min(v1[i], v2[i]));
        const tmax = @max(v0[i], @max(v1[i], v2[i]));
        if (tmin > h[i] or tmax < -h[i]) return false;
    }

    // 1 axis: the triangle plane normal.
    const n = vector.cross(edges[0], edges[1]);
    if (@abs(vector.dot(n, v0)) > vector.dot(h, @abs(n))) return false;

    return true;
}

/// Separating-axis test (Ericson RTCD, 15 axes) for two boxes. `rot` carries box B's orientation
/// expressed in box A's frame — only its 3x3 rotation is used, its columns being B's local axes in
/// A's frame — and `t` is B's centre relative to A's centre, in A's frame. `half_a` / `half_b` are
/// the boxes' half extents. `eps` is added to each |axis| to guard near-parallel edge cross products
/// whose length is (near) zero. Returns true when the boxes overlap. Ported from JoltPhysics OrientedBox.
fn sat_box_box(comptime T: type, rot: Mat(T, 4, 4), t: @Vector(3, T), half_a: @Vector(3, T), half_b: @Vector(3, T), eps: T) bool {
    // r[i][j] = rot(row i, col j) (column-major storage -> items[col][row]); column j is B's axis j.
    var r: [3][3]T = undefined;
    inline for (0..3) |i| {
        inline for (0..3) |j| {
            r[i][j] = rot.items[j][i];
        }
    }
    // absr[c][i] = |r[i][c]| + eps : component i of |B's axis c| plus the epsilon guard.
    var absr: [3][3]T = undefined;
    inline for (0..3) |c| {
        inline for (0..3) |i| {
            absr[c][i] = @abs(r[i][c]) + eps;
        }
    }

    // Test axes L = A0, A1, A2 (A's own axes).
    inline for (0..3) |i| {
        const ra = half_a[i];
        const rb = half_b[0] * absr[0][i] + half_b[1] * absr[1][i] + half_b[2] * absr[2][i];
        if (@abs(t[i]) > ra + rb) return false;
    }
    // Test axes L = B0, B1, B2 (B's own axes).
    inline for (0..3) |i| {
        const ra = half_a[0] * absr[i][0] + half_a[1] * absr[i][1] + half_a[2] * absr[i][2];
        const rb = half_b[i];
        if (@abs(t[0] * r[0][i] + t[1] * r[1][i] + t[2] * r[2][i]) > ra + rb) return false;
    }

    // Nine edge-edge cross-product axes L = Ai x Bj.
    if (@abs(t[2] * r[1][0] - t[1] * r[2][0]) > half_a[1] * absr[0][2] + half_a[2] * absr[0][1] + half_b[1] * absr[2][0] + half_b[2] * absr[1][0]) return false; // A0 x B0
    if (@abs(t[2] * r[1][1] - t[1] * r[2][1]) > half_a[1] * absr[1][2] + half_a[2] * absr[1][1] + half_b[0] * absr[2][0] + half_b[2] * absr[0][0]) return false; // A0 x B1
    if (@abs(t[2] * r[1][2] - t[1] * r[2][2]) > half_a[1] * absr[2][2] + half_a[2] * absr[2][1] + half_b[0] * absr[1][0] + half_b[1] * absr[0][0]) return false; // A0 x B2
    if (@abs(t[0] * r[2][0] - t[2] * r[0][0]) > half_a[0] * absr[0][2] + half_a[2] * absr[0][0] + half_b[1] * absr[2][1] + half_b[2] * absr[1][1]) return false; // A1 x B0
    if (@abs(t[0] * r[2][1] - t[2] * r[0][1]) > half_a[0] * absr[1][2] + half_a[2] * absr[1][0] + half_b[0] * absr[2][1] + half_b[2] * absr[0][1]) return false; // A1 x B1
    if (@abs(t[0] * r[2][2] - t[2] * r[0][2]) > half_a[0] * absr[2][2] + half_a[2] * absr[2][0] + half_b[0] * absr[1][1] + half_b[1] * absr[0][1]) return false; // A1 x B2
    if (@abs(t[1] * r[0][0] - t[0] * r[1][0]) > half_a[0] * absr[0][1] + half_a[1] * absr[0][0] + half_b[1] * absr[2][2] + half_b[2] * absr[1][2]) return false; // A2 x B0
    if (@abs(t[1] * r[0][1] - t[0] * r[1][1]) > half_a[0] * absr[1][1] + half_a[1] * absr[1][0] + half_b[0] * absr[2][2] + half_b[2] * absr[0][2]) return false; // A2 x B1
    if (@abs(t[1] * r[0][2] - t[0] * r[1][2]) > half_a[0] * absr[2][1] + half_a[1] * absr[2][0] + half_b[0] * absr[1][2] + half_b[1] * absr[0][2]) return false; // A2 x B2

    // No separating axis found.
    return true;
}

/// Oriented-box vs oriented-box overlap via the separating-axis theorem. `eps` guards near-parallel
/// edge cross products (JoltPhysics uses 1.0e-6 by default).
pub fn overlap_obb_obb(a: anytype, b: anytype, eps: @TypeOf(a).child) bool {
    comptime {
        std.debug.assert(@TypeOf(a).child == @TypeOf(b).child);
        std.debug.assert(@TypeOf(a).primative_type == .OrientedBox);
        std.debug.assert(@TypeOf(b).primative_type == .OrientedBox);
    }
    const T = @TypeOf(a).child;
    // Express B in A's frame: rot = inverse(A.orientation) * B.orientation.
    const rot = a.orientation.inverse_ortho().mul(b.orientation);
    return sat_box_box(T, rot, rot.position(), a.half_extent, b.half_extent, eps);
}

/// Oriented-box vs axis-aligned-box overlap via the separating-axis theorem. The AABB is treated as
/// box A (its frame is the world, so no rotation) and the OBB as box B expressed relative to the
/// AABB centre. `eps` guards near-parallel edge cross products.
pub fn overlap_obb_aabb(obb: anytype, aabb: anytype, eps: @TypeOf(obb).child) bool {
    comptime {
        std.debug.assert(@TypeOf(obb).child == @TypeOf(aabb).child);
        std.debug.assert(@TypeOf(obb).primative_type == .OrientedBox);
        std.debug.assert(@TypeOf(aabb).primative_type == .AABB);
    }
    const T = @TypeOf(obb).child;
    const t = obb.orientation.position() - aabb.get_center();
    return sat_box_box(T, obb.orientation, t, aabb.get_extent(), obb.half_extent, eps);
}

// generic overlap function
pub inline fn overlap(a: anytype, b: anytype) bool {
    const a_primative: geometry.Primative = @TypeOf(a).primative_type;
    const b_primative: geometry.Primative = @TypeOf(b).primative_type;
    if (a_primative == .Sphere and b_primative == .Sphere) return overlap_sphere_sphere(a, b);
    if (a_primative == .AABB and b_primative == .AABB) return overlap_aabb_aabb(a, b);
    if (a_primative == .AABB and b_primative == .Plane) return overlap_aabb_plane(a, b);
    if (a_primative == .Plane and b_primative == .AABB) return overlap_aabb_plane(b, a);
    if (a_primative == .AABB and b_primative == .Sphere) return overlap_aabb_sphere(a, b);
    if (a_primative == .Sphere and b_primative == .AABB) return overlap_aabb_sphere(b, a);
    if (a_primative == .OrientedBox and b_primative == .OrientedBox) return overlap_obb_obb(a, b, 1.0e-6);
    if (a_primative == .OrientedBox and b_primative == .AABB) return overlap_obb_aabb(a, b, 1.0e-6);
    if (a_primative == .AABB and b_primative == .OrientedBox) return overlap_obb_aabb(b, a, 1.0e-6);
    @compileError("Unsupported primative overlap: " ++ @tagName(a_primative) ++ " " ++ @tagName(b_primative));
}

test overlap_aabb_triangle {
    const box = geometry.AABB(f32).from_two_points(.{ -1, -1, -1 }, .{ 1, 1, 1 });
    // Triangle slicing through the box.
    try std.testing.expect(overlap_aabb_triangle(box, .{ -2, 0, 0 }, .{ 2, 0, 0 }, .{ 0, 2, 0 }));
    // Triangle with a vertex inside the box.
    try std.testing.expect(overlap_aabb_triangle(box, .{ 0, 0, 0 }, .{ 5, 0, 0 }, .{ 0, 5, 0 }));
    // Triangle far outside the box.
    try std.testing.expect(!overlap_aabb_triangle(box, .{ 5, 5, 5 }, .{ 6, 5, 5 }, .{ 5, 6, 5 }));
    // Triangle separated by a face plane (all z > 1).
    try std.testing.expect(!overlap_aabb_triangle(box, .{ 0, 0, 2 }, .{ 1, 0, 3 }, .{ 0, 1, 2.5 }));
}

test overlap_sphere_sphere {
    const s1: Sphere(f32) = .from_center_radius(.{ 0, 0, 0 }, 1);
    const s2: Sphere(f32) = .from_center_radius(.{ 0, 0, 1.5 }, 1);
    const s3: Sphere(f32) = .from_center_radius(.{ 0, 0, 3 }, 1);
    
    try std.testing.expect(overlap_sphere_sphere(s1, s2));
    try std.testing.expect(!overlap_sphere_sphere(s1, s3));
    // Symmetric
    try std.testing.expect(overlap(s1, s2));
    try std.testing.expect(overlap(s2, s1));
}

test overlap_aabb_aabb {
    const aabb1: geometry.AABB(f32) = .from_two_points(.{ 0, 0, 0 }, .{ 2, 2, 2 });
    const aabb2: geometry.AABB(f32) = .from_two_points(.{ 1, 1, 1 }, .{ 3, 3, 3 }); // Overlapping
    const aabb3: geometry.AABB(f32) = .from_two_points(.{ 5, 5, 5 }, .{ 7, 7, 7 }); // Non-overlapping
    const aabb4: geometry.AABB(f32) = .from_two_points(.{ 2, 0, 0 }, .{ 4, 2, 2 }); // Edge touching

    try std.testing.expect(overlap_aabb_aabb(aabb1, aabb2)); // Overlapping boxes should return true
    try std.testing.expect(!overlap_aabb_aabb(aabb1, aabb3)); // Non-overlapping boxes should return false
    try std.testing.expect(overlap_aabb_aabb(aabb1, aabb4)); // Edge touching boxes should return true


    // Symmetric - test the generic overlap function
    try std.testing.expect(overlap(aabb1, aabb2));
    try std.testing.expect(!overlap(aabb1, aabb3));
    try std.testing.expect(overlap(aabb2, aabb1));
}

test overlap_aabb_aabb_4 {
    const box: geometry.AABB(f32) = .from_two_points(.{ 0, 0, 0 }, .{ 2, 2, 2 });
    // Four candidate boxes packed component-wise: overlapping, far, edge-touching, far.
    const result = overlap_aabb_aabb_4(
        box,
        .{ 1, 5, 2, -5 }, // minX
        .{ 3, 7, 4, -3 }, // maxX
        .{ 1, 5, 0, -5 }, // minY
        .{ 3, 7, 2, -3 }, // maxY
        .{ 1, 5, 0, -5 }, // minZ
        .{ 3, 7, 2, -3 }, // maxZ
    );
    try std.testing.expectEqual(@Vector(4, bool){ true, false, true, false }, result);
}

test overlap_aabb_plane {
    const box: geometry.AABB(f32) = .from_two_points(.{ -1, -1, -1 }, .{ 1, 1, 1 });
    const through = geometry.Plane(f32).from_point_and_normal(.{ 0, 0, 0 }, .{ 0, 0, 1 });
    const far = geometry.Plane(f32).from_point_and_normal(.{ 0, 0, 5 }, .{ 0, 0, 1 });
    try std.testing.expect(overlap_aabb_plane(box, through));
    try std.testing.expect(!overlap_aabb_plane(box, far));
    // Routed through the generic dispatcher, both argument orders.
    try std.testing.expect(overlap(box, through));
    try std.testing.expect(overlap(through, box));
    try std.testing.expect(!overlap(box, far));
}

test overlap_aabb_sphere {
    const box: geometry.AABB(f32) = .from_two_points(.{ 0, 0, 0 }, .{ 2, 2, 2 });
    const near: Sphere(f32) = .from_center_radius(.{ 3, 1, 1 }, 1.5); // 1 unit from the face
    const far: Sphere(f32) = .from_center_radius(.{ 5, 1, 1 }, 1); // 3 units from the face
    try std.testing.expect(overlap_aabb_sphere(box, near));
    try std.testing.expect(!overlap_aabb_sphere(box, far));
    // Routed through the generic dispatcher, both argument orders.
    try std.testing.expect(overlap(box, near));
    try std.testing.expect(overlap(near, box));
    try std.testing.expect(!overlap(box, far));
}

test overlap_obb_obb {
    const OBB = geometry.OrientedBoundedBox(f32);
    const I = Mat(f32, 4, 4).identity;
    const unit: @Vector(3, f32) = .{ 1, 1, 1 };
    const a = OBB.from_orientation_and_half_extent(I, unit);
    // Axis-aligned: centres 1.5 apart (gap < 2) overlap; 3 apart (gap > 2) are separated.
    const b = OBB.from_orientation_and_half_extent(I.modify_column(3, @as(@Vector(3, f32), .{ 1.5, 0, 0 })), unit);
    const c = OBB.from_orientation_and_half_extent(I.modify_column(3, @as(@Vector(3, f32), .{ 3, 0, 0 })), unit);
    try std.testing.expect(overlap_obb_obb(a, b, 1.0e-6));
    try std.testing.expect(!overlap_obb_obb(a, c, 1.0e-6));
    // Generic dispatcher, both orders.
    try std.testing.expect(overlap(a, b));
    try std.testing.expect(overlap(b, a));

    // Rotated 45 deg about z reaches sqrt(2) ~ 1.414 along x, so combined reach ~2.414: overlap at
    // centre 2.3, separated at 2.6.
    const r45 = I.rotate(std.math.pi / 4.0, .{ 0, 0, 1 });
    const d = OBB.from_orientation_and_half_extent(r45.modify_column(3, @as(@Vector(3, f32), .{ 2.3, 0, 0 })), unit);
    const e = OBB.from_orientation_and_half_extent(r45.modify_column(3, @as(@Vector(3, f32), .{ 2.6, 0, 0 })), unit);
    try std.testing.expect(overlap_obb_obb(a, d, 1.0e-6));
    try std.testing.expect(!overlap_obb_obb(a, e, 1.0e-6));
}

test overlap_obb_aabb {
    const OBB = geometry.OrientedBoundedBox(f32);
    const I = Mat(f32, 4, 4).identity;
    const box: geometry.AABB(f32) = .from_two_points(.{ -1, -1, -1 }, .{ 1, 1, 1 });
    const near = OBB.from_orientation_and_half_extent(I.modify_column(3, @as(@Vector(3, f32), .{ 1.5, 0, 0 })), .{ 1, 1, 1 });
    const far = OBB.from_orientation_and_half_extent(I.modify_column(3, @as(@Vector(3, f32), .{ 3, 0, 0 })), .{ 1, 1, 1 });
    try std.testing.expect(overlap_obb_aabb(near, box, 1.0e-6));
    try std.testing.expect(!overlap_obb_aabb(far, box, 1.0e-6));
    // Generic dispatcher, both orders.
    try std.testing.expect(overlap(near, box));
    try std.testing.expect(overlap(box, near));
    try std.testing.expect(!overlap(box, far));
}
