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

pub fn overlap_aabb_aabb_4(a: anytype, minX: @Vector(4, @TypeOf(a).inner_type), maxX: @Vector(4, @TypeOf(a).inner_type), minY: @Vector(4, @TypeOf(a).inner_type), maxY: @Vector(4, @TypeOf(a).inner_type), minZ: @Vector(4, @TypeOf(a).inner_type), maxZ: @Vector(4, @TypeOf(a).inner_type)) @Vector(4, bool) {
    comptime {
        std.debug.assert(@TypeOf(a).primative_type == .AABB);
    }
    const box1_minx = @as(@Vector(4, @TypeOf(a).inner_type), @splat(a.min[0]));
    const box1_miny = @as(@Vector(4, @TypeOf(a).inner_type), @splat(a.min[1]));
    const box1_minz = @as(@Vector(4, @TypeOf(a).inner_type), @splat(a.min[2]));
    const box1_maxx = @as(@Vector(4, @TypeOf(a).inner_type), @splat(a.max[0]));
    const box1_maxy = @as(@Vector(4, @TypeOf(a).inner_type), @splat(a.max[1]));
    const box1_maxz = @as(@Vector(4, @TypeOf(a).inner_type), @splat(a.max[2]));

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

// generic overlap function
pub inline fn overlap(a: anytype, b: anytype) bool {
    const a_primative: geometry.Primative = @TypeOf(a).primative_type;
    const b_primative: geometry.Primative = @TypeOf(b).primative_type;
    if (a_primative == .Sphere and b_primative == .Sphere) return overlap_sphere_sphere(a, b);
    if (a_primative == .AABB and b_primative == .AABB) return overlap_aabb_aabb(a, b);
    @compileError("Unsupported primative overlap: " ++ @typeName(a_primative) ++ " " ++ @typeName(b_primative));
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
