const std = @import("std");
const zml = @import("../root.zig");
const geometry = zml.geom;

pub fn Sphere(comptime T: type) type {
    return struct {
        pub const child: type = T;
        pub const primative_type: geometry.Primative = .Sphere;

        pub const Self = @This();
        pub const empty: Self = .{
            .center = @Vector(3, T){0, 0, 0} ,
            .radius = 0,
        };

        center: @Vector(3, T),
        radius: T,

        pub fn from_center_radius(center: @Vector(3, T), radius: T) Sphere(T) {
            return .{
                .center = center,
                .radius = radius,
            };
        }

        pub fn translate(self: Self, translation: @Vector(3, T)) Self {
            return .{ .center = self.center + translation, .radius = self.radius };
        }

        /// The point on the sphere's surface furthest along `direction`.
        pub fn get_support(self: Self, direction: @Vector(3, T)) @Vector(3, T) {
            return self.center + zml.vec.normalize(direction) * @as(@Vector(3, T), @splat(self.radius));
        }

        /// Transform by a 4x4 matrix. The radius is scaled by the largest axis scale, giving a
        /// conservative bounding sphere under non-uniform scale.
        pub fn transform(self: Self, mat: zml.Mat(T, 4, 4)) Self {
            const new_center = mat.extract(3, 3).mul(zml.vec.to_mat(self.center)).column(0) + mat.position();
            return .{ .center = new_center, .radius = self.radius * @reduce(.Max, mat.get_scale()) };
        }
    };
}

test "sphere get_support" {
    const s: Sphere(f32) = .from_center_radius(.{ 0, 0, 0 }, 2);
    try std.testing.expect(zml.vec.is_close_default(s.get_support(.{ 1, 0, 0 }), .{ 2, 0, 0 }));
    try std.testing.expect(zml.vec.is_close_default(s.get_support(.{ 0, -1, 0 }), .{ 0, -2, 0 }));
}

test "sphere translate" {
    const s: Sphere(f32) = .from_center_radius(.{ 1, 2, 3 }, 2);
    const moved = s.translate(.{ 1, 1, 1 });
    try std.testing.expect(zml.vec.is_close_default(moved.center, .{ 2, 3, 4 }));
    try std.testing.expectEqual(@as(f32, 2), moved.radius);
}

test "sphere transform" {
    const s: Sphere(f32) = .from_center_radius(.{ 1, 0, 0 }, 2);
    // Pure translation: center moves, radius unchanged.
    const m = zml.Mat(f32, 4, 4).identity.translate(.{ 0, 5, 0 });
    const t = s.transform(m);
    try std.testing.expect(zml.vec.is_close_default(t.center, .{ 1, 5, 0 }));
    try std.testing.expectApproxEqRel(@as(f32, 2), t.radius, 1.0e-6);
    // Uniform scale: radius scales.
    const scaled = s.transform(zml.Mat(f32, 4, 4).identity.scale(.{ 3, 3, 3 }));
    try std.testing.expectApproxEqRel(@as(f32, 6), scaled.radius, 1.0e-6);
}
