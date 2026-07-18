//! Support-function wrappers for convex collision detection (GJK/EPA). Each type exposes
//! `get_support(direction) -> furthest point of the shape along direction` and a `child` scalar
//! type, so they compose and plug into `gjk.zig`. Any zml primitive with a (maximizing)
//! `get_support` — Sphere, OrientedBoundedBox, Capsule, Cylinder — can be wrapped directly.
//! Ported from JoltPhysics `Jolt/Geometry/ConvexSupport.h`.

const std = @import("std");
const zml = @import("../root.zig");

/// Wraps a convex object with a 4x4 transform (rotation + translation + uniform scale).
pub fn TransformedConvexObject(comptime Object: type) type {
    return struct {
        const Self = @This();
        pub const child = Object.child;
        const T = child;

        transform: zml.Mat(T, 4, 4),
        object: Object,

        pub fn get_support(self: Self, direction: @Vector(3, T)) @Vector(3, T) {
            const r3 = self.transform.extract(3, 3);
            // Direction into the object's local frame: transpose(R) * direction.
            const local_dir = r3.transpose().mul(zml.vec.to_mat(direction)).column(0);
            const local = self.object.get_support(local_dir);
            // Support back into world: R * local + translation.
            return r3.mul(zml.vec.to_mat(local)).column(0) + self.transform.position();
        }
    };
}

/// Wraps a convex object and inflates it by a convex radius (rounds it off, e.g. turns a point into
/// a sphere or a box into a rounded box).
pub fn AddConvexRadius(comptime Object: type) type {
    return struct {
        const Self = @This();
        pub const child = Object.child;
        const T = child;

        object: Object,
        radius: T,

        pub fn get_support(self: Self, direction: @Vector(3, T)) @Vector(3, T) {
            const len = zml.vec.norm(direction);
            const base = self.object.get_support(direction);
            if (len > 0) return base + direction * @as(@Vector(3, T), @splat(self.radius / len));
            return base;
        }
    };
}

/// Minkowski difference A - B: the support hull GJK operates on. `get_support(d)` is
/// `A.get_support(d) - B.get_support(-d)`.
pub fn MinkowskiDifference(comptime A: type, comptime B: type) type {
    return struct {
        const Self = @This();
        pub const child = A.child;
        const T = child;

        object_a: A,
        object_b: B,

        pub fn get_support(self: Self, direction: @Vector(3, T)) @Vector(3, T) {
            return self.object_a.get_support(direction) - self.object_b.get_support(-direction);
        }
    };
}

/// A single point as a (degenerate) convex object; its support is always the point.
pub fn PointConvexSupport(comptime T: type) type {
    return struct {
        const Self = @This();
        pub const child = T;

        point: @Vector(3, T),

        pub fn get_support(self: Self, direction: @Vector(3, T)) @Vector(3, T) {
            _ = direction;
            return self.point;
        }
    };
}

/// A triangle as a convex object; its support is the vertex furthest along the direction.
pub fn TriangleConvexSupport(comptime T: type) type {
    return struct {
        const Self = @This();
        pub const child = T;

        v1: @Vector(3, T),
        v2: @Vector(3, T),
        v3: @Vector(3, T),

        pub fn get_support(self: Self, direction: @Vector(3, T)) @Vector(3, T) {
            const d1 = zml.vec.dot(self.v1, direction);
            const d2 = zml.vec.dot(self.v2, direction);
            const d3 = zml.vec.dot(self.v3, direction);
            if (d1 > d2) return if (d1 > d3) self.v1 else self.v3;
            return if (d2 > d3) self.v2 else self.v3;
        }
    };
}

/// A convex polygon (or point cloud) given as a slice of vertices; its support is the vertex
/// furthest along the direction.
pub fn PolygonConvexSupport(comptime T: type) type {
    return struct {
        const Self = @This();
        pub const child = T;

        vertices: []const @Vector(3, T),

        pub fn get_support(self: Self, direction: @Vector(3, T)) @Vector(3, T) {
            var best = self.vertices[0];
            var best_dot = zml.vec.dot(self.vertices[0], direction);
            for (self.vertices[1..]) |v| {
                const d = zml.vec.dot(v, direction);
                if (d > best_dot) {
                    best_dot = d;
                    best = v;
                }
            }
            return best;
        }
    };
}

test TransformedConvexObject {
    // A point at local origin, translated to (5, 0, 0): support is always (5, 0, 0).
    const p = PointConvexSupport(f32){ .point = .{ 0, 0, 0 } };
    const moved = TransformedConvexObject(PointConvexSupport(f32)){
        .transform = zml.Mat(f32, 4, 4).identity.translate(.{ 5, 0, 0 }),
        .object = p,
    };
    try std.testing.expect(zml.vec.is_close_default(moved.get_support(.{ 1, 0, 0 }), .{ 5, 0, 0 }));
    try std.testing.expect(zml.vec.is_close_default(moved.get_support(.{ 0, 1, 0 }), .{ 5, 0, 0 }));
}

test AddConvexRadius {
    // Point at origin inflated by radius 2 behaves like a sphere.
    const p = PointConvexSupport(f32){ .point = .{ 0, 0, 0 } };
    const sphere = AddConvexRadius(PointConvexSupport(f32)){ .object = p, .radius = 2 };
    try std.testing.expect(zml.vec.is_close_default(sphere.get_support(.{ 1, 0, 0 }), .{ 2, 0, 0 }));
    try std.testing.expect(zml.vec.is_close_default(sphere.get_support(.{ 0, -3, 0 }), .{ 0, -2, 0 }));
    // Zero direction is handled without a divide-by-zero.
    try std.testing.expect(zml.vec.is_close_default(sphere.get_support(.{ 0, 0, 0 }), .{ 0, 0, 0 }));
}

test MinkowskiDifference {
    // Difference of two points is their difference vector, regardless of direction.
    const a = PointConvexSupport(f32){ .point = .{ 3, 0, 0 } };
    const b = PointConvexSupport(f32){ .point = .{ 1, 0, 0 } };
    const diff = MinkowskiDifference(PointConvexSupport(f32), PointConvexSupport(f32)){ .object_a = a, .object_b = b };
    try std.testing.expect(zml.vec.is_close_default(diff.get_support(.{ 1, 0, 0 }), .{ 2, 0, 0 }));
    try std.testing.expect(zml.vec.is_close_default(diff.get_support(.{ -1, 0, 0 }), .{ 2, 0, 0 }));
}

test TriangleConvexSupport {
    const tri = TriangleConvexSupport(f32){ .v1 = .{ 0, 0, 0 }, .v2 = .{ 2, 0, 0 }, .v3 = .{ 0, 2, 0 } };
    try std.testing.expect(zml.vec.is_close_default(tri.get_support(.{ 1, 0, 0 }), .{ 2, 0, 0 }));
    try std.testing.expect(zml.vec.is_close_default(tri.get_support(.{ 0, 1, 0 }), .{ 0, 2, 0 }));
    try std.testing.expect(zml.vec.is_close_default(tri.get_support(.{ -1, -1, 0 }), .{ 0, 0, 0 }));
}
