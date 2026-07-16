const matrix = @import("matrix.zig");
const std = @import("std");

pub const vec = @import("vector.zig");
pub const quat = @import("quat.zig");
pub const Mat = matrix.Mat;

pub const Vec4f32 = vec.Vec4f32;
pub const Vec3f32 = vec.Vec3f32;
pub const Vec2f32 = vec.Vec2f32;

pub const Vec4f64 = vec.Vec4f64;
pub const Vec3f64 = vec.Vec3f64;
pub const Vec2f64 = vec.Vec2f64;

pub const Quat4f32 = quat.Quat4f32;
pub const Quat4f64 = quat.Quat4f64;

pub const Mat4f32 = matrix.Mat4f32;
pub const Mat4f64 = matrix.Mat4f64;

pub const Mat3f32 = Mat(f32, 3, 3);
pub const Mat3f64 = Mat(f64, 3, 3);

// Affine transform matrices with the implicit last row elided: an N x (N+1) matrix that
// represents an (N+1) x (N+1) transform. Row-major, so byte-compatible with a compact GPU
// transform (translation in each row's last element).
pub const Mat2x3f32 = Mat(f32, 2, 3);
pub const Mat3x4f32 = Mat(f32, 3, 4);

pub const geom = @import("geometry.zig");
pub const meta = @import("meta.zig");
pub const simd = @import("simd.zig");
pub const scalar = @import("scalar.zig");
pub const packing = @import("packing.zig");
pub const color = @import("color.zig");
pub const random = @import("random.zig");
pub const testing = @import("testing.zig");
pub const approx = @import("approx.zig");
pub const bivec = @import("bivec.zig");
pub const rotor = @import("rotor.zig");
pub const affine = @import("affine.zig");
pub const spring = @import("spring.zig");

test {
    _ = @import("vector.zig");
    _ = @import("matrix.zig");
    _ = @import("quat.zig");
    _ = @import("geometry.zig");
    _ = @import("meta.zig");
    _ = @import("simd.zig");
    _ = @import("scalar.zig");
    _ = @import("packing.zig");
    _ = @import("color.zig");
    _ = @import("random.zig");
    _ = @import("testing.zig");
    _ = @import("approx.zig");
    _ = @import("bivec.zig");
    _ = @import("rotor.zig");
    _ = @import("affine.zig");
    _ = @import("spring.zig");
}

// NOTE: bivec/rotor/affine/spring are being ported incrementally; each must exist for this file
// to compile. They are created in dependency order (bivec -> rotor -> affine, spring).

pub fn to_radians(degrees: anytype) @TypeOf(degrees) {
    return degrees * (std.math.pi / 180);
}

pub fn to_degrees(radians: anytype) @TypeOf(radians) {
    return radians * (1 / (std.math.pi / 180));
}

pub fn find_roots(comptime T: type, a: T, b: T, c: T) struct {
    num_roots: u8,
    roots: [2]T,
} {
    // Check if this is a linear equation
    if (a == 0) {
        // Check if this is a constant equation
        if (b == 0)
            return .{
                .num_roots = 0,
                .roots = .{ 0, 0 },
            };

        // Linear equation with 1 solution
        const r1 = -c / b;
        return .{
            .num_roots = 1,
            .roots = .{ r1, r1 },
        };
    }

    // See Numerical Recipes in C, Chapter 5.6 Quadratic and Cubic Equations
    const det: T = (b * b) - 4 * a * c;
    if (det < 0)
        return .{
            .num_roots = 0,
            .roots = .{ 0, 0 },
        };

    const q: T = (b + std.math.sign(b) * std.math.sqrt(det)) / -2;
    const r1 = q / a;
    if (q == 0) {
        return .{
            .num_roots = 1,
            .roots = .{ r1, r1 },
        };
    }
    const r2 = c / q;
    return .{
        .num_roots = 2,
        .roots = .{ r1, r2 },
    };
}
