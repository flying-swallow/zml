const std = @import("std");
const zml = @import("../root.zig");
const Mat = @import("../matrix.zig").Mat;
const vec = @import("../vector.zig");
const geometry = @import("../geometry.zig");
const Plane = @import("plane.zig").Plane;

/// A view frustum represented by its six bounding planes with inward-pointing
/// normals: left, right, bottom, top, near, far. Built from a view-projection
/// matrix using the Gribb-Hartmann method for a right-handed, [0, 1]-depth
/// projection (the convention produced by `Mat.perspective`).
pub fn Frustum(comptime T: type) type {
    return struct {
        pub const child: type = T;
        pub const primative_type: geometry.Primative = .Frustum;
        const Self = @This();

        planes: [6]Plane(T),

        fn make_plane(v: @Vector(4, T)) Plane(T) {
            const n = @Vector(3, T){ v[0], v[1], v[2] };
            const inv_len = 1.0 / vec.norm(n);
            return .{
                .normal = n * @as(@Vector(3, T), @splat(inv_len)),
                .c = v[3] * inv_len,
            };
        }

        /// Extract the six frustum planes from a (model-)view-projection matrix.
        pub fn from_view_projection(mvp: Mat(T, 4, 4)) Self {
            const r0 = mvp.row(0);
            const r1 = mvp.row(1);
            const r2 = mvp.row(2);
            const r3 = mvp.row(3);
            return .{
                .planes = .{
                    make_plane(r3 + r0), // left
                    make_plane(r3 - r0), // right
                    make_plane(r3 + r1), // bottom
                    make_plane(r3 - r1), // top
                    make_plane(r2), // near (zero-to-one depth)
                    make_plane(r3 - r2), // far
                },
            };
        }

        pub fn intersect_sphere_state(self: Self, sphere: anytype) geometry.IntersectionState {
            comptime std.debug.assert(@TypeOf(sphere).primative_type == .Sphere);
            var inside = true;
            for (self.planes) |p| {
                const d = p.signed_distance(sphere.center);
                if (d < -sphere.radius) return .outside;
                if (d < sphere.radius) inside = false;
            }
            return if (inside) .inside else .partial;
        }

        pub fn intersect_sphere(self: Self, sphere: anytype) bool {
            return self.intersect_sphere_state(sphere) != .outside;
        }

        pub fn intersect_aabb_state(self: Self, aabb: anytype) geometry.IntersectionState {
            comptime std.debug.assert(@TypeOf(aabb).primative_type == .AABB);
            var inside = true;
            for (self.planes) |p| {
                // AABB.get_support(d) returns the corner minimizing dot(d, corner),
                // so get_support(-normal) is the corner furthest along +normal.
                if (p.signed_distance(aabb.get_support(-p.normal)) < 0) return .outside;
                if (p.signed_distance(aabb.get_support(p.normal)) < 0) inside = false;
            }
            return if (inside) .inside else .partial;
        }

        pub fn intersect_aabb(self: Self, aabb: anytype) bool {
            return self.intersect_aabb_state(aabb) != .outside;
        }

        pub fn intersect_capsule_state(self: Self, capsule: anytype) geometry.IntersectionState {
            comptime std.debug.assert(@TypeOf(capsule).primative_type == .Capsule);
            var inside = true;
            for (self.planes) |p| {
                const d0 = p.signed_distance(capsule.hemisphere_centers[0]);
                const d1 = p.signed_distance(capsule.hemisphere_centers[1]);
                if (@max(d0, d1) < -capsule.radius) return .outside;
                if (@min(d0, d1) < capsule.radius) inside = false;
            }
            return if (inside) .inside else .partial;
        }

        pub fn intersect_capsule(self: Self, capsule: anytype) bool {
            return self.intersect_capsule_state(capsule) != .outside;
        }
    };
}

test "frustum culling" {
    const M = Mat(f32, 4, 4).orthographic(-1, 1, -1, 1, 1, 10);
    const f = Frustum(f32).from_view_projection(M);
    const Sphere = zml.geom.Sphere;
    const AABB = zml.geom.AABB;
    const Capsule = zml.geom.Capsule;
    const State = geometry.IntersectionState;

    // Spheres
    try std.testing.expectEqual(State.inside, f.intersect_sphere_state(Sphere(f32).from_center_radius(.{ 0, 0, -5 }, 0.1)));
    try std.testing.expectEqual(State.outside, f.intersect_sphere_state(Sphere(f32).from_center_radius(.{ 5, 0, -5 }, 0.1)));
    try std.testing.expectEqual(State.outside, f.intersect_sphere_state(Sphere(f32).from_center_radius(.{ 0, 0, -0.5 }, 0.1)));
    try std.testing.expectEqual(State.partial, f.intersect_sphere_state(Sphere(f32).from_center_radius(.{ 1, 0, -5 }, 0.5)));
    try std.testing.expect(f.intersect_sphere(Sphere(f32).from_center_radius(.{ 0, 0, -5 }, 0.1)));
    try std.testing.expect(!f.intersect_sphere(Sphere(f32).from_center_radius(.{ 5, 0, -5 }, 0.1)));

    // AABBs
    try std.testing.expectEqual(State.inside, f.intersect_aabb_state(AABB(f32).from_two_points(.{ -0.5, -0.5, -6 }, .{ 0.5, 0.5, -4 })));
    try std.testing.expectEqual(State.outside, f.intersect_aabb_state(AABB(f32).from_two_points(.{ 2, 2, -6 }, .{ 3, 3, -4 })));
    try std.testing.expectEqual(State.partial, f.intersect_aabb_state(AABB(f32).from_two_points(.{ 0.5, 0.5, -6 }, .{ 1.5, 1.5, -4 })));

    // Capsules
    try std.testing.expect(f.intersect_capsule(Capsule(f32){ .hemisphere_centers = .{ .{ 0, 0, -4 }, .{ 0, 0, -6 } }, .radius = 0.2 }));
    try std.testing.expect(!f.intersect_capsule(Capsule(f32){ .hemisphere_centers = .{ .{ 5, 0, -4 }, .{ 5, 0, -6 } }, .radius = 0.2 }));
}
