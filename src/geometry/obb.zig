const std = @import("std");
const Mat = @import("../matrix.zig").Mat;
const geometry = @import("../geometry.zig");
const zml = @import("../root.zig");

pub fn OrientedBoundedBox(comptime T: type) type {
    return struct {
        pub const primative_type = geometry.Primative.OrientedBox;
        pub const child: type = T;
        const Self = @This();

        orientation: Mat(T, 4, 4), // rotation + translation of the box in world space
        half_extent: @Vector(3, T), // half the box size along each local axis

        pub fn from_orientation_and_half_extent(orientation: Mat(T, 4, 4), half_extent: @Vector(3, T)) Self {
            return .{ .orientation = orientation, .half_extent = half_extent };
        }

        /// World-space centre of the box (the translation of the orientation matrix).
        pub fn get_center(self: Self) @Vector(3, T) {
            return self.orientation.position();
        }

        /// The corner of the box furthest along `direction` in world space.
        pub fn get_support(self: Self, direction: @Vector(3, T)) @Vector(3, T) {
            var support = self.orientation.position();
            inline for (0..3) |i| {
                const col = self.orientation.column(i); // world-space local axis i (unit if unscaled)
                const axis = @Vector(3, T){ col[0], col[1], col[2] };
                const s: T = if (zml.vec.dot(direction, axis) >= 0) self.half_extent[i] else -self.half_extent[i];
                support += axis * @as(@Vector(3, T), @splat(s));
            }
            return support;
        }

        /// Transform by a rigid 4x4 matrix (rotation + translation). The half extents are
        /// unchanged, so a matrix with scale is not handled here.
        pub fn transform(self: Self, mat: Mat(T, 4, 4)) Self {
            return .{ .orientation = mat.mul(self.orientation), .half_extent = self.half_extent };
        }
    };
}

test "obb get_center" {
    const obb = OrientedBoundedBox(f32).from_orientation_and_half_extent(
        Mat(f32, 4, 4).identity.translate(.{ 1, 2, 3 }),
        .{ 1, 1, 1 },
    );
    try std.testing.expect(zml.vec.is_close_default(obb.get_center(), .{ 1, 2, 3 }));
}

test "obb get_support axis aligned" {
    const obb = OrientedBoundedBox(f32).from_orientation_and_half_extent(
        Mat(f32, 4, 4).identity,
        .{ 1, 2, 3 },
    );
    try std.testing.expect(zml.vec.is_close_default(obb.get_support(.{ 1, 1, 1 }), .{ 1, 2, 3 }));
    try std.testing.expect(zml.vec.is_close_default(obb.get_support(.{ -1, -1, -1 }), .{ -1, -2, -3 }));
    try std.testing.expect(zml.vec.is_close_default(obb.get_support(.{ 1, -1, 1 }), .{ 1, -2, 3 }));
}

test "obb get_support rotated" {
    // Box rotated 90 degrees about z: local +x -> world +y, local +y -> world -x.
    const obb = OrientedBoundedBox(f32).from_orientation_and_half_extent(
        Mat(f32, 4, 4).identity.rotate(std.math.pi / 2.0, .{ 0, 0, 1 }),
        .{ 2, 1, 1 },
    );
    // Direction picking +local-x, +local-y, +local-z -> world corner (-1, 2, 1).
    try std.testing.expect(zml.vec.is_close_default(obb.get_support(.{ -1, 1, 1 }), .{ -1, 2, 1 }));
}

test "obb transform" {
    const obb = OrientedBoundedBox(f32).from_orientation_and_half_extent(
        Mat(f32, 4, 4).identity,
        .{ 1, 1, 1 },
    );
    const moved = obb.transform(Mat(f32, 4, 4).identity.translate(.{ 4, -1, 2 }));
    try std.testing.expect(zml.vec.is_close_default(moved.get_center(), .{ 4, -1, 2 }));
}
