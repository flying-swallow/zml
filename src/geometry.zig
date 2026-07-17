const std = @import("std");
const vec = @import("vector.zig");

pub const Primative = enum {
    AABB,
    Plane,
    Sphere,
    OrientedBox,
    Capsule,
    Frustum,
    Cylinder,
    Segment,
};

/// Result of testing a primitive against a bounding volume such as a frustum.
pub const IntersectionState = enum {
    outside,
    inside,
    partial,
};

pub const AABB = @import("geometry/aabb.zig").AABB;
pub const AABBf32 = AABB(f32);
pub const AABBf64 = AABB(f64);

pub const Plane = @import("geometry/plane.zig").Plane;
pub const Planef32 = Plane(f32);
pub const Planef64 = Plane(f64);

pub const Sphere = @import("geometry/sphere.zig").Sphere;
pub const Capsule = @import("geometry/capsule.zig").Capsule;
pub const OrientedBoundedBox = @import("geometry/obb.zig").OrientedBoundedBox;
pub const Frustum = @import("geometry/frustum.zig").Frustum;
pub const Frustumf32 = Frustum(f32);
pub const Frustumf64 = Frustum(f64);

pub const Cylinder = @import("geometry/cylinder.zig").Cylinder;
pub const Segment = @import("geometry/segment.zig").Segment;



pub const overlap = @import("geometry/overlap.zig");
pub const ray = @import("geometry/ray.zig");
pub const contains = @import("geometry/contains.zig");

test {
    _ = @import("geometry/aabb.zig");
    _ = @import("geometry/plane.zig");
    _ = @import("geometry/sphere.zig");
    _ = @import("geometry/capsule.zig");
    _ = @import("geometry/obb.zig");
    _ = @import("geometry/frustum.zig");
    _ = @import("geometry/overlap.zig");
    _ = @import("geometry/ray.zig");
    _ = @import("geometry/contains.zig");
    _ = @import("geometry/cylinder.zig");
    _ = @import("geometry/segment.zig");
}

/// Read a fixed-size vector (or array) `T` out of a packed byte buffer. `vertex_index`
/// selects the element, `byte_offset` skips a header or attribute offset, and `byte_stride`
/// is the distance in bytes between consecutive elements (use the tightly-packed size for
/// contiguous data, a larger value for interleaved vertex attributes).
pub fn get_vector_from_buffer(comptime T: type, vertex_index: usize, buffer: []align(4) const u8, byte_offset: usize, byte_stride: usize) T {
    const vector_len = switch (@typeInfo(T)) {
        .vector => |v| v.len,
        .array => |a| a.len,
        else => @compileError("Expected a vector or array type, got: " ++ @typeName(T)),
    };
    var arr: [vector_len]std.meta.Child(T) = undefined;
    const total_bytes = vector_len * @sizeOf(std.meta.Child(T));
    const begin = byte_offset + vertex_index * byte_stride;
    @memcpy(std.mem.asBytes(&arr), buffer[begin .. begin + total_bytes]);
    return arr;
}

test "get_vector_from_buffer - tightly packed 2D f32" {
    const data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const buffer = std.mem.sliceAsBytes(data[0..]);
    try std.testing.expectEqual(@Vector(2, f32){ 1, 2 }, get_vector_from_buffer(@Vector(2, f32), 0, buffer, 0, 2 * @sizeOf(f32)));
    try std.testing.expectEqual(@Vector(2, f32){ 5, 6 }, get_vector_from_buffer(@Vector(2, f32), 2, buffer, 0, 2 * @sizeOf(f32)));
}

test "get_vector_from_buffer - interleaved with offset and stride" {
    // Two vertices, each position(3) + normal(3); stride is 6 floats.
    const data = [_]f32{
        1, 2, 3, 0.1, 0.2, 0.3,
        4, 5, 6, 0.4, 0.5, 0.6,
    };
    const buffer = std.mem.sliceAsBytes(data[0..]);
    const stride = 6 * @sizeOf(f32);
    try std.testing.expectEqual(@Vector(3, f32){ 4, 5, 6 }, get_vector_from_buffer(@Vector(3, f32), 1, buffer, 0, stride));
    const n0 = get_vector_from_buffer(@Vector(3, f32), 0, buffer, 3 * @sizeOf(f32), stride);
    try std.testing.expect(vec.is_close_default(n0, @Vector(3, f32){ 0.1, 0.2, 0.3 }));
}

test "get_vector_from_buffer - array type" {
    const data = [_]i32{ 1, 2, 3, 4 };
    const buffer = std.mem.sliceAsBytes(data[0..]);
    try std.testing.expectEqual([2]i32{ 3, 4 }, get_vector_from_buffer([2]i32, 1, buffer, 0, 2 * @sizeOf(i32)));
}
