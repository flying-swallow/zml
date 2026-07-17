const std = @import("std");
const geometry = @import("../geometry.zig");
const zml = @import("../root.zig");

pub fn Capsule(comptime T: type) type {
    return struct {
        pub const child: type = T;
        pub const primative_type = geometry.Primative.Capsule;
        const Self = @This();

        hemisphere_centers: [2]@Vector(3, T), // the two hemisphere centers
        radius: T, // radius of the capsule

        pub fn center(self: Self) @Vector(3, T) {
            return (self.hemisphere_centers[0] + self.hemisphere_centers[1]) * @as(@Vector(3, T), @splat(0.5));
        }

        pub fn get_cylinder_height(self: Self) T {
            return zml.vec.distance(self.hemisphere_centers[0], self.hemisphere_centers[1]);
        }

        pub fn get_total_height(self: Self) T {
            return self.get_cylinder_height() + self.radius * @as(T, 2);
        }

        /// The point on the capsule's surface furthest along `direction`: the hemisphere
        /// centre furthest along `direction`, offset by the radius.
        pub fn get_support(self: Self, direction: @Vector(3, T)) @Vector(3, T) {
            const d0 = zml.vec.dot(self.hemisphere_centers[0], direction);
            const d1 = zml.vec.dot(self.hemisphere_centers[1], direction);
            const c = if (d1 > d0) self.hemisphere_centers[1] else self.hemisphere_centers[0];
            return c + zml.vec.normalize(direction) * @as(@Vector(3, T), @splat(self.radius));
        }

        /// Transform by a 4x4 matrix. The radius is scaled by the largest axis scale.
        pub fn transform(self: Self, mat: zml.Mat(T, 4, 4)) Self {
            const r3 = mat.extract(3, 3);
            const t = mat.position();
            return .{
                .hemisphere_centers = .{
                    r3.mul(zml.vec.to_mat(self.hemisphere_centers[0])).column(0) + t,
                    r3.mul(zml.vec.to_mat(self.hemisphere_centers[1])).column(0) + t,
                },
                .radius = self.radius * @reduce(.Max, mat.get_scale()),
            };
        }
    };
}

test "capsule get_support" {
    const c: zml.geom.Capsule(f32) = .{ .hemisphere_centers = .{ .{ 0, 0, -1 }, .{ 0, 0, 1 } }, .radius = 1 };
    // Along +z: top hemisphere centre pushed out by the radius.
    try std.testing.expect(zml.vec.is_close_default(c.get_support(.{ 0, 0, 1 }), .{ 0, 0, 2 }));
    // Along +x: either centre (tie), pushed out along +x.
    try std.testing.expect(zml.vec.is_close_default(c.get_support(.{ 1, 0, 0 }), .{ 1, 0, -1 }));
}

test "capsule transform" {
    const c: zml.geom.Capsule(f32) = .{ .hemisphere_centers = .{ .{ 0, 0, -1 }, .{ 0, 0, 1 } }, .radius = 1 };
    const m = zml.Mat(f32, 4, 4).identity.translate(.{ 2, 0, 0 });
    const t = c.transform(m);
    try std.testing.expect(zml.vec.is_close_default(t.hemisphere_centers[0], .{ 2, 0, -1 }));
    try std.testing.expect(zml.vec.is_close_default(t.hemisphere_centers[1], .{ 2, 0, 1 }));
    try std.testing.expectApproxEqRel(@as(f32, 1), t.radius, 1.0e-6);
}

