const std = @import("std");

const INV_2_32: f32 = 2.3283064365386963e-10; // 1 / 2^32

// ---------------------------------------------------------------------------
// Integer hashing / stateless PRNG steps
// ---------------------------------------------------------------------------

/// Wellons' "lowbias32" integer hash — a strong bit mixer for u32 keys.
pub fn hash_u32(x0: u32) u32 {
    var x = x0;
    x ^= x >> 16;
    x *%= 0x7feb352d;
    x ^= x >> 15;
    x *%= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

/// Fold a new value into a running hash (boost-style combiner).
pub fn hash_combine(seed: u32, v: u32) u32 {
    return seed ^ (hash_u32(v) +% 0x9e3779b9 +% (seed << 6) +% (seed >> 2));
}

/// Numerical Recipes linear congruential step.
pub fn lcg_next(state: u32) u32 {
    return state *% 1664525 +% 1013904223;
}

/// Marsaglia xorshift32 step (state must be non-zero).
pub fn xorshift32(state: u32) u32 {
    var x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

/// Map a u32 to a float in [0, 1) using the top 24 bits.
pub fn uint_to_float01(u: u32) f32 {
    return @as(f32, @floatFromInt(u >> 8)) * (1.0 / 16777216.0);
}

// ---------------------------------------------------------------------------
// Low-discrepancy sequences
// ---------------------------------------------------------------------------

/// The van der Corput / Halton radical inverse of `index` in the given base.
pub fn halton(index: u32, base: u32) f32 {
    var f: f32 = 1;
    var r: f32 = 0;
    var i = index;
    const bf: f32 = @floatFromInt(base);
    while (i > 0) {
        f /= bf;
        r += f * @as(f32, @floatFromInt(i % base));
        i /= base;
    }
    return r;
}

fn radical_inverse_base2(bits0: u32) f32 {
    var bits = bits0;
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);
    return @as(f32, @floatFromInt(bits)) * INV_2_32;
}

/// The i-th of N points of the Hammersley set on the unit square.
pub fn hammersley_2d(i: u32, n: u32) @Vector(2, f32) {
    return .{ @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n)), radical_inverse_base2(i) };
}

/// Weyl additive recurrence in 1D (golden-ratio increment).
pub fn weyl_1d(n: u32) f32 {
    return @as(f32, @floatFromInt(n *% 2654435769)) * INV_2_32;
}

/// The R2 low-discrepancy sequence (Roberts 2018), a 2D generalization of the
/// golden-ratio sequence.
pub fn weyl_2d(n: u32) @Vector(2, f32) {
    return .{
        @as(f32, @floatFromInt(n *% 3242174889)) * INV_2_32,
        @as(f32, @floatFromInt(n *% 2447445414)) * INV_2_32,
    };
}

// ---------------------------------------------------------------------------
// Morton / Z-order curve (16 bits per axis)
// ---------------------------------------------------------------------------

fn part1by1(x0: u32) u32 {
    var x = x0 & 0x0000FFFF;
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    return x;
}

fn compact1by1(x0: u32) u32 {
    var x = x0 & 0x55555555;
    x = (x | (x >> 1)) & 0x33333333;
    x = (x | (x >> 2)) & 0x0F0F0F0F;
    x = (x | (x >> 4)) & 0x00FF00FF;
    x = (x | (x >> 8)) & 0x0000FFFF;
    return x;
}

pub fn morton2d_encode(x: u16, y: u16) u32 {
    return part1by1(x) | (part1by1(y) << 1);
}

pub fn morton2d_decode(code: u32) struct { x: u16, y: u16 } {
    return .{ .x = @intCast(compact1by1(code)), .y = @intCast(compact1by1(code >> 1)) };
}

test "hash determinism" {
    try std.testing.expectEqual(hash_u32(42), hash_u32(42));
    try std.testing.expect(hash_u32(1) != hash_u32(2));
    try std.testing.expect(hash_combine(0, 5) != hash_combine(0, 6));
}

test "uint_to_float01 range" {
    try std.testing.expectEqual(@as(f32, 0), uint_to_float01(0));
    const f = uint_to_float01(0xFFFFFFFF);
    try std.testing.expect(f >= 0 and f < 1);
}

test "halton base 2" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), halton(1, 2), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), halton(2, 2), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.75), halton(3, 2), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), halton(1, 3), 1e-6);
}

test "hammersley/weyl range" {
    try std.testing.expectEqual(@Vector(2, f32){ 0, 0 }, hammersley_2d(0, 16));
    const w = weyl_2d(7);
    try std.testing.expect(w[0] >= 0 and w[0] < 1 and w[1] >= 0 and w[1] < 1);
}

test "morton round trip" {
    const code = morton2d_encode(12345, 54321);
    const d = morton2d_decode(code);
    try std.testing.expectEqual(@as(u16, 12345), d.x);
    try std.testing.expectEqual(@as(u16, 54321), d.y);
    // Interleaving: (1,0) -> bit0, (0,1) -> bit1.
    try std.testing.expectEqual(@as(u32, 1), morton2d_encode(1, 0));
    try std.testing.expectEqual(@as(u32, 2), morton2d_encode(0, 1));
}
