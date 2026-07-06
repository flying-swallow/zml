const std = @import("std");

// ---------------------------------------------------------------------------
// Bit reinterpretation
// ---------------------------------------------------------------------------

pub fn asuint(x: f32) u32 {
    return @bitCast(x);
}

pub fn asfloat(x: u32) f32 {
    return @bitCast(x);
}

// ---------------------------------------------------------------------------
// UNORM / SNORM  (fixed-point normalized integers of arbitrary bit width)
// ---------------------------------------------------------------------------

/// Pack a value in [0, 1] into a `bits`-wide unsigned normalized integer.
pub fn pack_unorm(comptime bits: u16, x: f32) u32 {
    const maxv: f32 = @floatFromInt((@as(u64, 1) << bits) - 1);
    return @intFromFloat(@round(std.math.clamp(x, 0, 1) * maxv));
}

/// Unpack a `bits`-wide unsigned normalized integer back into [0, 1].
pub fn unpack_unorm(comptime bits: u16, v: u32) f32 {
    const maxv: f32 = @floatFromInt((@as(u64, 1) << bits) - 1);
    return @as(f32, @floatFromInt(v)) / maxv;
}

/// Pack a value in [-1, 1] into a `bits`-wide signed normalized integer.
pub fn pack_snorm(comptime bits: u16, x: f32) i32 {
    const maxv: f32 = @floatFromInt((@as(u64, 1) << (bits - 1)) - 1);
    return @intFromFloat(@round(std.math.clamp(x, -1, 1) * maxv));
}

/// Unpack a `bits`-wide signed normalized integer back into [-1, 1].
pub fn unpack_snorm(comptime bits: u16, v: i32) f32 {
    const maxv: f32 = @floatFromInt((@as(u64, 1) << (bits - 1)) - 1);
    return @max(@as(f32, @floatFromInt(v)) / maxv, -1.0);
}

// ---------------------------------------------------------------------------
// Half precision (IEEE binary16). Zig has a native f16, so these are bit casts.
// ---------------------------------------------------------------------------

pub fn pack_f16(x: f32) u16 {
    return @bitCast(@as(f16, @floatCast(x)));
}

pub fn unpack_f16(v: u16) f32 {
    return @floatCast(@as(f16, @bitCast(v)));
}

// ---------------------------------------------------------------------------
// Packed float formats derived from the binary16 layout (5-bit exponent, bias 15)
// ---------------------------------------------------------------------------

/// R11G11B10F: 11-bit (5e6m) R and G, 10-bit (5e5m) B, no sign. Negatives clamp to 0.
/// Mantissa bits are truncated, so this is a lossy pack.
pub fn pack_ufloat_11_11_10(rgb: @Vector(3, f32)) u32 {
    const r: u32 = f32_to_ufloat(rgb[0], 6);
    const g: u32 = f32_to_ufloat(rgb[1], 6);
    const b: u32 = f32_to_ufloat(rgb[2], 5);
    return r | (g << 11) | (b << 22);
}

pub fn unpack_ufloat_11_11_10(v: u32) @Vector(3, f32) {
    return .{
        ufloat_to_f32(v & 0x7FF, 6),
        ufloat_to_f32((v >> 11) & 0x7FF, 6),
        ufloat_to_f32((v >> 22) & 0x3FF, 5),
    };
}

fn f32_to_ufloat(x: f32, comptime mantissa_bits: u4) u32 {
    if (!(x > 0)) return 0; // <= 0 or NaN
    const h: u16 = @bitCast(@as(f16, @floatCast(x)));
    const exp: u32 = (h >> 10) & 0x1F;
    const mant: u32 = (h >> (10 - mantissa_bits)) & ((@as(u32, 1) << mantissa_bits) - 1);
    return (exp << mantissa_bits) | mant;
}

fn ufloat_to_f32(v: u32, comptime mantissa_bits: u4) f32 {
    const exp: u16 = @intCast((v >> mantissa_bits) & 0x1F);
    const mant: u16 = @intCast(v & ((@as(u32, 1) << mantissa_bits) - 1));
    const h: u16 = (exp << 10) | (mant << (10 - mantissa_bits));
    return @floatCast(@as(f16, @bitCast(h)));
}

// ---------------------------------------------------------------------------
// FP8 (OCP E5M2): 1 sign, 5 exponent, 2 mantissa — the top 8 bits of binary16.
// ---------------------------------------------------------------------------

pub fn pack_f8_e5m2(x: f32) u8 {
    const h: u16 = @bitCast(@as(f16, @floatCast(x)));
    return @intCast(h >> 8);
}

pub fn unpack_f8_e5m2(v: u8) f32 {
    return @floatCast(@as(f16, @bitCast(@as(u16, v) << 8)));
}

test "unorm/snorm round trip" {
    for ([_]f32{ 0, 0.25, 0.5, 0.75, 1 }) |x| {
        try std.testing.expectApproxEqAbs(x, unpack_unorm(8, pack_unorm(8, x)), 1.0 / 255.0);
        try std.testing.expectApproxEqAbs(x, unpack_unorm(16, pack_unorm(16, x)), 1.0 / 65535.0);
    }
    for ([_]f32{ -1, -0.5, 0, 0.5, 1 }) |x| {
        try std.testing.expectApproxEqAbs(x, unpack_snorm(8, pack_snorm(8, x)), 1.0 / 127.0);
        try std.testing.expectApproxEqAbs(x, unpack_snorm(16, pack_snorm(16, x)), 1.0 / 32767.0);
    }
    // Out-of-range values saturate.
    try std.testing.expectEqual(@as(f32, 1), unpack_unorm(8, pack_unorm(8, 5)));
    try std.testing.expectEqual(@as(f32, 0), unpack_unorm(8, pack_unorm(8, -5)));
}

test "f16 round trip" {
    for ([_]f32{ 0, 1, -2, 0.5, 100.25 }) |x| {
        try std.testing.expectEqual(x, unpack_f16(pack_f16(x)));
    }
}

test "asfloat/asuint" {
    try std.testing.expectEqual(@as(f32, 1.5), asfloat(asuint(1.5)));
    try std.testing.expectEqual(@as(u32, 0x3F800000), asuint(1.0));
}

test "ufloat_11_11_10 round trip" {
    // Values with exact low-order mantissa survive the pack.
    const rgb = @Vector(3, f32){ 1.0, 2.0, 0.5 };
    const back = unpack_ufloat_11_11_10(pack_ufloat_11_11_10(rgb));
    try std.testing.expectApproxEqAbs(rgb[0], back[0], 1e-6);
    try std.testing.expectApproxEqAbs(rgb[1], back[1], 1e-6);
    try std.testing.expectApproxEqAbs(rgb[2], back[2], 1e-6);
    // Negatives clamp to zero.
    const neg = unpack_ufloat_11_11_10(pack_ufloat_11_11_10(.{ -1, 0, 0 }));
    try std.testing.expectEqual(@as(f32, 0), neg[0]);
}

test "f8_e5m2 round trip" {
    for ([_]f32{ 0, 0.5, 1, 1.5, 2, -2 }) |x| {
        try std.testing.expectEqual(x, unpack_f8_e5m2(pack_f8_e5m2(x)));
    }
}
