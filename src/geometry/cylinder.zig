const std = @import("std");
const geometry = @import("../geometry.zig");
const zml = @import("../root.zig");

pub fn Cylinder(comptime T: type) type {
    return struct {
        pub const child: type = T;
        pub const primative_type = geometry.Primative.Cylinder;
        const Self = @This();

        end_points: [2]@Vector(3, T), // the two end-cap centers
        radius: T, // radius of the cylinder

        pub fn from_two_points_radius(p0: @Vector(3, T), p1: @Vector(3, T), radius: T) Self {
            return .{ .end_points = .{ p0, p1 }, .radius = radius };
        }

        pub fn center(self: Self) @Vector(3, T) {
            return (self.end_points[0] + self.end_points[1]) * @as(@Vector(3, T), @splat(0.5));
        }

        pub fn get_height(self: Self) T {
            return zml.vec.distance(self.end_points[0], self.end_points[1]);
        }

        /// Unit vector along the cylinder axis (from end_points[0] to end_points[1]).
        pub fn axis(self: Self) @Vector(3, T) {
            return zml.vec.normalize(self.end_points[1] - self.end_points[0]);
        }

        /// Support point of the capped cylinder in the given direction: the cap furthest
        /// along `direction`, pushed out to the rim by the radial part of `direction`.
        pub fn get_support(self: Self, direction: @Vector(3, T)) @Vector(3, T) {
            const u = self.axis();
            const axial = zml.vec.dot(direction, u);
            const cap = if (axial >= 0) self.end_points[1] else self.end_points[0];
            const radial = direction - u * @as(@Vector(3, T), @splat(axial));
            const radial_len = zml.vec.norm(radial);
            if (radial_len <= 1.0e-12) return cap;
            return cap + radial * @as(@Vector(3, T), @splat(self.radius / radial_len));
        }
    };
}

test "cylinder basics" {
    const c = Cylinder(f32).from_two_points_radius(.{ 0, 0, 0 }, .{ 0, 0, 2 }, 1.0);
    try std.testing.expect(zml.vec.is_close_default(c.center(), .{ 0, 0, 1 }));
    try std.testing.expectApproxEqRel(@as(f32, 2), c.get_height(), 1.0e-6);
    try std.testing.expect(zml.vec.is_close_default(c.axis(), .{ 0, 0, 1 }));
}

test "cylinder get_support" {
    const c = Cylinder(f32).from_two_points_radius(.{ 0, 0, 0 }, .{ 0, 0, 2 }, 1.0);
    // +x: rim of the top cap on the +x side.
    try std.testing.expect(zml.vec.is_close_default(c.get_support(.{ 1, 0, 0 }), .{ 1, 0, 2 }));
    // straight down the axis: bottom cap center (no radial component).
    try std.testing.expect(zml.vec.is_close_default(c.get_support(.{ 0, 0, -1 }), .{ 0, 0, 0 }));
}
