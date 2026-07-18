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

        /// Grow the sphere just enough to contain `point`. A point already inside is a no-op.
        /// Adapted from JoltPhysics Sphere::EncapsulatePoint.
        pub fn encapsulate_point(self: Self, point: @Vector(3, T)) Self {
            const d_vec = point - self.center;
            const d_sq = zml.vec.norm_sqr(d_vec);
            if (d_sq <= self.radius * self.radius) return self;
            const d = @sqrt(d_sq);
            const radius = 0.5 * (self.radius + d);
            return .{
                .center = self.center + d_vec * @as(@Vector(3, T), @splat((radius - self.radius) / d)),
                .radius = radius,
            };
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

test "sphere encapsulate_point" {
    const s: Sphere(f32) = .from_center_radius(.{ 0, 0, 0 }, 1);
    // Point inside: sphere unchanged.
    const in = s.encapsulate_point(.{ 0.5, 0, 0 });
    try std.testing.expect(zml.vec.is_close_default(in.center, .{ 0, 0, 0 }));
    try std.testing.expectApproxEqRel(@as(f32, 1), in.radius, 1.0e-6);
    // Point outside at x = 3: new sphere spans x in [-1, 3] -> center (1,0,0), radius 2.
    const out = s.encapsulate_point(.{ 3, 0, 0 });
    try std.testing.expect(zml.vec.is_close_default(out.center, .{ 1, 0, 0 }));
    try std.testing.expectApproxEqRel(@as(f32, 2), out.radius, 1.0e-6);
    // The far point lies on the enlarged surface and the original sphere is still contained.
    try std.testing.expectApproxEqRel(@as(f32, 2), zml.vec.distance(out.center, .{ 3, 0, 0 }), 1.0e-6);
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
