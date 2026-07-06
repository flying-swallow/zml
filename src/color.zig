const std = @import("std");

const Vec3 = @Vector(3, f32);

fn map3(v: Vec3, comptime f: fn (f32) f32) Vec3 {
    return .{ f(v[0]), f(v[1]), f(v[2]) };
}

// ---------------------------------------------------------------------------
// sRGB transfer functions (IEC 61966-2-1)
// ---------------------------------------------------------------------------

pub fn srgb_to_linear(c: f32) f32 {
    return if (c <= 0.04045) c / 12.92 else std.math.pow(f32, (c + 0.055) / 1.055, 2.4);
}

pub fn linear_to_srgb(c: f32) f32 {
    return if (c <= 0.0031308) c * 12.92 else 1.055 * std.math.pow(f32, c, 1.0 / 2.4) - 0.055;
}

pub fn srgb_to_linear3(c: Vec3) Vec3 {
    return map3(c, srgb_to_linear);
}

pub fn linear_to_srgb3(c: Vec3) Vec3 {
    return map3(c, linear_to_srgb);
}

// ---------------------------------------------------------------------------
// Luminance (Rec. 709 linear-light weights)
// ---------------------------------------------------------------------------

pub fn luminance(rgb: Vec3) f32 {
    return @reduce(.Add, rgb * Vec3{ 0.2126, 0.7152, 0.0722 });
}

// ---------------------------------------------------------------------------
// RGB <-> YCoCg (reversible luma/chroma transform)
// ---------------------------------------------------------------------------

pub fn rgb_to_ycocg(rgb: Vec3) Vec3 {
    return .{
        0.25 * rgb[0] + 0.5 * rgb[1] + 0.25 * rgb[2],
        0.5 * rgb[0] - 0.5 * rgb[2],
        -0.25 * rgb[0] + 0.5 * rgb[1] - 0.25 * rgb[2],
    };
}

pub fn ycocg_to_rgb(ycocg: Vec3) Vec3 {
    const y = ycocg[0];
    const co = ycocg[1];
    const cg = ycocg[2];
    return .{ y + co - cg, y + cg, y - co - cg };
}

// ---------------------------------------------------------------------------
// Tone mapping
// ---------------------------------------------------------------------------

pub fn reinhard(x: Vec3) Vec3 {
    return x / (@as(Vec3, @splat(1)) + x);
}

pub fn reinhard_inverse(x: Vec3) Vec3 {
    return x / (@as(Vec3, @splat(1)) - x);
}

/// Narkowicz's fitted ACES filmic tone mapping curve.
pub fn aces_film(x: Vec3) Vec3 {
    const a: Vec3 = @splat(2.51);
    const b: Vec3 = @splat(0.03);
    const c: Vec3 = @splat(2.43);
    const d: Vec3 = @splat(0.59);
    const e: Vec3 = @splat(0.14);
    const num = x * (a * x + b);
    const den = x * (c * x + d) + e;
    return @min(@max(num / den, @as(Vec3, @splat(0))), @as(Vec3, @splat(1)));
}

// ---------------------------------------------------------------------------
// PQ / SMPTE ST 2084 (Rec. 2100). Linear input normalized so 1.0 == 10000 nits.
// ---------------------------------------------------------------------------

const pq_m1 = 0.1593017578125;
const pq_m2 = 78.84375;
const pq_c1 = 0.8359375;
const pq_c2 = 18.8515625;
const pq_c3 = 18.6875;

pub fn linear_to_pq(l: f32) f32 {
    const lm1 = std.math.pow(f32, @max(l, 0), pq_m1);
    return std.math.pow(f32, (pq_c1 + pq_c2 * lm1) / (1.0 + pq_c3 * lm1), pq_m2);
}

pub fn pq_to_linear(n: f32) f32 {
    const nm2 = std.math.pow(f32, @max(n, 0), 1.0 / pq_m2);
    return std.math.pow(f32, @max(nm2 - pq_c1, 0) / (pq_c2 - pq_c3 * nm2), 1.0 / pq_m1);
}

test "srgb round trip" {
    for ([_]f32{ 0, 0.001, 0.04, 0.5, 1 }) |c| {
        try std.testing.expectApproxEqAbs(c, linear_to_srgb(srgb_to_linear(c)), 1e-5);
    }
    const v = Vec3{ 0.1, 0.5, 0.9 };
    const back = linear_to_srgb3(srgb_to_linear3(v));
    try std.testing.expectApproxEqAbs(v[0], back[0], 1e-5);
    try std.testing.expectApproxEqAbs(v[2], back[2], 1e-5);
}

test "luminance" {
    try std.testing.expectApproxEqAbs(@as(f32, 1), luminance(.{ 1, 1, 1 }), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7152), luminance(.{ 0, 1, 0 }), 1e-6);
}

test "ycocg round trip" {
    const rgb = Vec3{ 0.2, 0.6, 0.9 };
    const back = ycocg_to_rgb(rgb_to_ycocg(rgb));
    try std.testing.expectApproxEqAbs(rgb[0], back[0], 1e-6);
    try std.testing.expectApproxEqAbs(rgb[1], back[1], 1e-6);
    try std.testing.expectApproxEqAbs(rgb[2], back[2], 1e-6);
}

test "reinhard round trip" {
    const x = Vec3{ 0.1, 1.0, 4.0 };
    const back = reinhard_inverse(reinhard(x));
    try std.testing.expectApproxEqAbs(x[0], back[0], 1e-5);
    try std.testing.expectApproxEqAbs(x[2], back[2], 1e-4);
}

test "pq round trip" {
    for ([_]f32{ 0.0, 0.01, 0.1, 0.5, 1.0 }) |l| {
        try std.testing.expectApproxEqAbs(l, pq_to_linear(linear_to_pq(l)), 1e-4);
    }
}
