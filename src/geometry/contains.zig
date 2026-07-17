const std = @import("std");
const zml = @import("../root.zig");

pub fn aabb_contains_point(a: anytype, pt: @Vector(3, @TypeOf(a).child)) bool {
    comptime {
        std.debug.assert(@TypeOf(a).primative_type == .AABB);
    }
    return @reduce(.And, (pt >= a.min) & (pt <= a.max));
}

pub fn capsule_contains_point(a: anytype, pt: @Vector(3, @TypeOf(a).child)) bool {
    comptime {
        std.debug.assert(@TypeOf(a).primative_type == .Capsule);
    }
    const ab = a.hemisphere_centers[1] - a.hemisphere_centers[0];
    const t = @max(@min(zml.vec.dot(pt - a.hemisphere_centers[0], ab) / zml.vec.dot(ab, ab), @as(@TypeOf(a).child, 1)), @as(@TypeOf(a).child, 0));
    const closest_point = a.hemisphere_centers[0] + ab * @as(@Vector(3, @TypeOf(a).child), @splat(t));
    return zml.vec.distance_sqr(pt, closest_point) <= a.radius * a.radius;
}

pub fn sphere_contains_point(a: anytype, pt: @Vector(3, @TypeOf(a).child)) bool {
    comptime {
        std.debug.assert(@TypeOf(a).primative_type == .Sphere);
    }
    return zml.vec.distance_sqr(pt, a.center) <= a.radius * a.radius;
}

pub fn obb_contains_point(a: anytype, pt: @Vector(3, @TypeOf(a).child)) bool {
    comptime {
        std.debug.assert(@TypeOf(a).primative_type == .OrientedBox);
    }
    const T = @TypeOf(a).child;
    // Project the offset from the box centre onto each (unit) local axis; the point is
    // inside iff every projection lies within the corresponding half extent.
    const d = pt - a.orientation.position();
    inline for (0..3) |i| {
        const col = a.orientation.column(i);
        const axis = @Vector(3, T){ col[0], col[1], col[2] };
        if (@abs(zml.vec.dot(d, axis)) > a.half_extent[i]) return false;
    }
    return true;
}

/// Generic point-containment dispatcher, routing on the primitive's `primative_type`.
pub fn contains_point(a: anytype, pt: @Vector(3, @TypeOf(a).child)) bool {
    const p = @TypeOf(a).primative_type;
    if (p == .AABB) return aabb_contains_point(a, pt);
    if (p == .Sphere) return sphere_contains_point(a, pt);
    if (p == .Capsule) return capsule_contains_point(a, pt);
    if (p == .OrientedBox) return obb_contains_point(a, pt);
    @compileError("Unsupported primative contains_point: " ++ @tagName(p));
}

test aabb_contains_point {
    const aabb: zml.geom.AABB(f32) = .from_two_points(.{ -1, -1, -1 }, .{ 1, 1, 1 });
    try std.testing.expect(aabb_contains_point(aabb, .{ 0, 0, 0 }));
    try std.testing.expect(!aabb_contains_point(aabb, .{ 2, 0, 0 }));
    try std.testing.expect(!aabb_contains_point(aabb, .{ 0, -2, 0 }));
    try std.testing.expect(!aabb_contains_point(aabb, .{ 0, 0, 2 }));
}

test capsule_contains_point {
    const capsule: zml.geom.Capsule(f32) = .{
        .hemisphere_centers = .{ .{ 0, 0, -1 }, .{ 0, 0, 1 } },
        .radius = 1,
    };
    try std.testing.expect(capsule_contains_point(capsule, .{ 0, 0, 0 }));
    try std.testing.expect(capsule_contains_point(capsule, .{ 1, 0, 0 }));
    try std.testing.expect(!capsule_contains_point(capsule, .{ 2, 0, 0 }));
    try std.testing.expect(!capsule_contains_point(capsule, .{ 0, -2, 0 }));
    try std.testing.expect(!capsule_contains_point(capsule, .{ 0, 0, 3 }));
}

test sphere_contains_point {
    const sphere: zml.geom.Sphere(f32) = .from_center_radius(.{ 0, 0, 0 }, 2);
    try std.testing.expect(sphere_contains_point(sphere, .{ 1, 0, 0 }));
    try std.testing.expect(sphere_contains_point(sphere, .{ 0, 0, 2 }));
    try std.testing.expect(!sphere_contains_point(sphere, .{ 3, 0, 0 }));
}

test obb_contains_point {
    // Box rotated 90 degrees about z, half extents (2, 1, 1), centred at (1, 0, 0).
    const obb: zml.geom.OrientedBoundedBox(f32) = .from_orientation_and_half_extent(
        zml.Mat(f32, 4, 4).identity.rotate(std.math.pi / 2.0, .{ 0, 0, 1 }).translate(.{ 1, 0, 0 }),
        .{ 2, 1, 1 },
    );
    try std.testing.expect(obb_contains_point(obb, .{ 1, 0, 0 })); // centre
    // Local +x extends 2 units along world +y, so (1, 1.9, 0) is inside but (2.1, 0, 0) is not.
    try std.testing.expect(obb_contains_point(obb, .{ 1, 1.9, 0 }));
    try std.testing.expect(!obb_contains_point(obb, .{ 1, 2.1, 0 }));
    try std.testing.expect(!obb_contains_point(obb, .{ 2.1, 0, 0 }));
}

test contains_point {
    const aabb: zml.geom.AABB(f32) = .from_two_points(.{ -1, -1, -1 }, .{ 1, 1, 1 });
    const sphere: zml.geom.Sphere(f32) = .from_center_radius(.{ 0, 0, 0 }, 2);
    const capsule: zml.geom.Capsule(f32) = .{ .hemisphere_centers = .{ .{ 0, 0, -1 }, .{ 0, 0, 1 } }, .radius = 1 };
    const obb: zml.geom.OrientedBoundedBox(f32) = .from_orientation_and_half_extent(zml.Mat(f32, 4, 4).identity, .{ 1, 1, 1 });

    try std.testing.expect(contains_point(aabb, .{ 0, 0, 0 }));
    try std.testing.expect(!contains_point(aabb, .{ 2, 0, 0 }));
    try std.testing.expect(contains_point(sphere, .{ 1, 1, 0 }));
    try std.testing.expect(!contains_point(sphere, .{ 3, 0, 0 }));
    try std.testing.expect(contains_point(capsule, .{ 1, 0, 0 }));
    try std.testing.expect(!contains_point(capsule, .{ 2, 0, 0 }));
    try std.testing.expect(contains_point(obb, .{ 0.5, 0.5, 0.5 }));
    try std.testing.expect(!contains_point(obb, .{ 2, 0, 0 }));
}
