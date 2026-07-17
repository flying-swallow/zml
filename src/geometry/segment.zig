const std = @import("std");
const geometry = @import("../geometry.zig");
const zml = @import("../root.zig");

pub fn Segment(comptime T: type) type {
    return struct {
        pub const child: type = T;
        pub const primative_type = geometry.Primative.Segment;
        const Self = @This();

        start: @Vector(3, T),
        end: @Vector(3, T),

        pub fn from_two_points(start: @Vector(3, T), end: @Vector(3, T)) Self {
            return .{ .start = start, .end = end };
        }

        pub fn direction(self: Self) @Vector(3, T) {
            return self.end - self.start;
        }

        pub fn length(self: Self) T {
            return zml.vec.distance(self.start, self.end);
        }

        pub fn length_sqr(self: Self) T {
            return zml.vec.distance_sqr(self.start, self.end);
        }

        pub fn center(self: Self) @Vector(3, T) {
            return (self.start + self.end) * @as(@Vector(3, T), @splat(0.5));
        }

        /// Closest point on the segment to `pt`, clamped to the endpoints.
        pub fn closest_point(self: Self, pt: @Vector(3, T)) @Vector(3, T) {
            const ab = self.end - self.start;
            const denom = zml.vec.dot(ab, ab);
            if (denom <= 0) return self.start; // degenerate segment (start == end)
            const t = @max(@min(zml.vec.dot(pt - self.start, ab) / denom, @as(T, 1)), @as(T, 0));
            return self.start + ab * @as(@Vector(3, T), @splat(t));
        }
    };
}

test "segment basics" {
    const s = Segment(f32).from_two_points(.{ 0, 0, 0 }, .{ 0, 0, 4 });
    try std.testing.expectApproxEqRel(@as(f32, 4), s.length(), 1.0e-6);
    try std.testing.expectApproxEqRel(@as(f32, 16), s.length_sqr(), 1.0e-6);
    try std.testing.expect(zml.vec.is_close_default(s.center(), .{ 0, 0, 2 }));
    try std.testing.expect(zml.vec.is_close_default(s.direction(), .{ 0, 0, 4 }));
}

test "segment closest_point" {
    const s = Segment(f32).from_two_points(.{ 0, 0, 0 }, .{ 0, 0, 4 });
    // Beside the middle -> projects onto the middle.
    try std.testing.expect(zml.vec.is_close_default(s.closest_point(.{ 1, 0, 2 }), .{ 0, 0, 2 }));
    // Beyond the far end -> clamps to end.
    try std.testing.expect(zml.vec.is_close_default(s.closest_point(.{ 0, 0, 10 }), .{ 0, 0, 4 }));
    // Before the start -> clamps to start.
    try std.testing.expect(zml.vec.is_close_default(s.closest_point(.{ 0, 0, -3 }), .{ 0, 0, 0 }));
}
