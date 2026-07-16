//! Damped spring integration. A `DampedSpringMat` holds the four coefficients that advance a
//! (position, velocity) pair one timestep toward a target under a damped spring, and can be reused
//! across many springs that share `dt`/`freq`/`ratio`.
//!
//! The math is written out by hand rather than reusing `Mat` -- it keeps both the implementation
//! and the call sites readable and allows an extra `@mulAdd` in `apply`. Ported verbatim from
//! Games-by-Mason's mr_tween (MIT), which is based on Ryan Juckett's derivation:
//! https://www.ryanjuckett.com/damped-springs/
//!
//! The generic `.vector` branch mr_tween carried does not compile on Zig 0.17 (taking `&v[i]` on a
//! `*@Vector` yields a vector-element pointer that will not coerce to `*f32`), so it is dropped
//! here. Callers with `@Vector` data (e.g. the rotor spring) bridge through a stack array, which
//! hits the byte-identical `.array` branch.

const std = @import("std");
const approx = @import("approx.zig");
const zml_testing = @import("testing.zig");

const inv_exp = approx.inv_exp;
const log2_near_one = approx.log2_near_one;
const exp2_near_zero = approx.exp2_near_zero;
const assert = std.debug.assert;

/// A matrix of damped spring coefficients for a scalar float type `T`.
pub fn DampedSpringMat(T: type) type {
    return struct {
        d0_d0: T,
        d0_d1: T,
        d1_d0: T,
        d1_d1: T,

        /// The infinity matrix. When applied, instantly brings the spring to equilibrium.
        pub const inf: @This() = .{
            .d0_d0 = 0,
            .d0_d1 = 0,
            .d1_d0 = 0,
            .d1_d1 = 0,
        };

        pub const InitOptions = struct {
            /// The delta time since the last spring update.
            dt: T,
            /// The angular frequency of the spring ignoring damping.
            freq: T,
            /// The damping ratio.
            ///
            /// ratio < 1 is underdamped, will overshoot and bounce back
            /// ratio = 1 is critically damped, max speed without overshoot
            /// ratio > 1 is overdamped, slower than necessary to prevent overshoot
            ratio: T,
        };

        /// Initialize the damped spring matrix for the given spring parameters and delta time. If
        /// you have multiple springs with the same parameters, reuse this result rather than
        /// recomputing it for each spring.
        pub fn init(options: InitOptions) @This() {
            // We assume ratio and frequency are non-negative going forward
            const ratio = @max(options.ratio, 0);
            const freq = options.freq;
            const dt = options.dt;

            // If no time has passed, return the identity matrix
            if (dt <= 0 or freq <= 0) {
                return identity(dt);
            }

            // If the ratio is greater than 1, we're over damped
            if (ratio > 1.0) {
                const za = -freq * ratio;
                const zb = freq * @sqrt(@mulAdd(T, ratio, ratio, -1));
                const z1 = za - zb;
                const z2 = za + zb;

                // We can use our inverse exp function since z1 and z2 will always be negative
                const e1 = inv_exp(-z1 * dt);
                const e2 = inv_exp(-z2 * dt);

                const inv_two_zb = 0.5 / zb;

                const e1_over = e1 * inv_two_zb;
                const e2_over = e2 * inv_two_zb;

                const ze1_over = z1 * e1_over;
                const ze2_over = z2 * e2_over;

                const e2_minus_ze2_over = e2 - ze2_over;

                return .{
                    .d0_d0 = @mulAdd(T, e1_over, z2, e2_minus_ze2_over),
                    .d0_d1 = e2_over - e1_over,
                    .d1_d0 = (ze1_over + e2_minus_ze2_over) * z2,
                    .d1_d1 = ze2_over - ze1_over,
                };
            }

            // If the ratio is less than one, we're under damped
            if (ratio < 1.0) {
                const omega_zeta = freq * ratio;
                const alpha = freq * @sqrt(@mulAdd(T, -ratio, ratio, 1));

                const alpha_dt = alpha * dt;

                // We can use inv_exp since omega zeta will never be negative. We set cosine and
                // sine to 0 when we have an infinite alpha dt to avoid nan, these values will be
                // multiplied by an exp of zero momentarily anyway.
                const exp = inv_exp(omega_zeta * dt);
                const cos = if (std.math.isInf(alpha_dt)) 0 else @cos(alpha_dt);
                const sin = if (std.math.isInf(alpha_dt)) 0 else @sin(alpha_dt);

                const inv_alpha = 1 / alpha;

                const exp_sin = exp * sin;
                const exp_cos = exp * cos;
                const exp_o = exp * omega_zeta * sin * inv_alpha;

                return .{
                    .d0_d0 = exp_cos + exp_o,
                    .d0_d1 = exp_sin * inv_alpha,
                    .d1_d0 = @mulAdd(T, -exp_sin, alpha, -omega_zeta * exp_o),
                    .d1_d1 = exp_cos - exp_o,
                };
            }

            // Otherwise the ratio is exactly one and we're critically damped
            {
                // We can use inv exp since freq and dt will never be negative
                const exp = inv_exp(freq * dt);

                // We special case infinite dt to avoid multiplying zero by infinity and getting nan.
                const time_exp = if (std.math.isInf(dt)) 0 else dt * exp;
                const time_exp_freq = time_exp * freq;

                return .{
                    .d0_d0 = time_exp_freq + exp,
                    .d0_d1 = time_exp,
                    .d1_d0 = -freq * time_exp_freq,
                    .d1_d1 = exp - time_exp_freq,
                };
            }
        }

        /// Returns a matrix that applies no new spring force for the given timestep.
        pub fn identity(dt: T) @This() {
            return .{
                .d0_d0 = 1,
                .d0_d1 = dt,
                .d1_d0 = 0,
                .d1_d1 = 1,
            };
        }

        pub fn ApplyOptions(V: type) type {
            return struct {
                /// The current value. Typically interpreted as a position.
                d0: *V,
                /// The current derivative. Typically interpreted as a velocity.
                d1: *V,
                /// The target value. Typically interpreted as a position.
                target: V,
            };
        }

        /// Apply the damped spring matrix to the given position and velocity with the given target.
        /// If given a struct or array, the matrix is applied component wise to each value.
        ///
        /// Inputs are unchanged for the identity matrix. For any other matrix, if any input is NaN
        /// both d0 and d1 are set to NaN. Otherwise if any value (other than the matrix) is inf, the
        /// result is unspecified.
        pub fn apply(self: @This(), V: type, options: ApplyOptions(V)) void {
            switch (@typeInfo(V)) {
                .float, .comptime_float => {
                    const d0_prev = options.d0.* - options.target;
                    const d1_prev = options.d1.*;
                    options.d0.* = @mulAdd(
                        T,
                        d0_prev,
                        self.d0_d0,
                        @mulAdd(T, d1_prev, self.d0_d1, options.target),
                    );
                    options.d1.* = @mulAdd(
                        T,
                        d0_prev,
                        self.d1_d0,
                        d1_prev * self.d1_d1,
                    );
                },
                .array => |info| for (0..info.len) |i| {
                    self.apply(T, .{
                        .d0 = &options.d0[i],
                        .d1 = &options.d1[i],
                        .target = options.target[i],
                    });
                },
                .@"struct" => |info| inline for (info.field_names) |field_name| {
                    self.apply(T, .{
                        .d0 = &@field(options.d0, field_name),
                        .d1 = &@field(options.d1, field_name),
                        .target = @field(options.target, field_name),
                    });
                },
                else => comptime unreachable,
            }
        }

        pub fn ApplyOriginOptions(V: type) type {
            return struct {
                /// The current value. Typically interpreted as a position.
                d0: *V,
                /// The current derivative. Typically interpreted as a velocity.
                d1: *V,
            };
        }

        /// Similar to `apply`, but assumes a target of the origin which mildly simplifies the math.
        /// This can be useful when using a spring to minimize error, e.g. for rotations.
        pub fn apply_origin(self: @This(), V: type, options: ApplyOriginOptions(V)) void {
            switch (@typeInfo(V)) {
                .float, .comptime_float => {
                    const d0_prev = options.d0.*;
                    const d1_prev = options.d1.*;
                    options.d0.* = @mulAdd(
                        T,
                        d0_prev,
                        self.d0_d0,
                        d1_prev * self.d0_d1,
                    );
                    options.d1.* = @mulAdd(
                        T,
                        d0_prev,
                        self.d1_d0,
                        d1_prev * self.d1_d1,
                    );
                },
                .array => |info| for (0..info.len) |i| {
                    self.apply_origin(T, .{
                        .d0 = &options.d0[i],
                        .d1 = &options.d1[i],
                    });
                },
                .@"struct" => |info| inline for (info.field_names) |field_name| {
                    self.apply_origin(T, .{
                        .d0 = &@field(options.d0, field_name),
                        .d1 = &@field(options.d1, field_name),
                    });
                },
                else => comptime unreachable,
            }
        }

        pub fn ApplyExpOptions(V: type) type {
            const F = comp_type(V);
            return struct {
                /// The current value. Typically interpreted as a position.
                d0: *V,
                /// The current derivative. Typically interpreted as a velocity.
                d1: *V,
                /// The target value. Typically interpreted as a position.
                target: V,
                /// Use a fast approximation for log/exp when within this distance of 1/0.
                eps: F = if (F == comptime_float) 0 else std.math.floatEps(F),
            };
        }

        /// Similar to `apply`, but accepts exponential values. Useful for animating things like
        /// scale that are exponential in nature. Velocity measures how many times the value doubles
        /// per unit time.
        ///
        /// d0 is clamped both as an input and as an output to keep it greater than 0.
        pub fn apply_exp(self: @This(), V: type, options: ApplyExpOptions(V)) void {
            switch (@typeInfo(V)) {
                .float, .comptime_float => {
                    // Return exact results at t=0
                    if (std.meta.eql(self, identity(0))) return;

                    // Make sure d0 isn't too close to 0
                    if (@typeInfo(V) != .comptime_float and !std.math.isNan(options.d0.*)) {
                        options.d0.* = @max(options.d0.*, std.math.floatEps(f32));
                    }

                    // Simulate the spring. In the common case where the spring has settled, we'll
                    // use the fast approximate ln/exp.
                    var dist = log2_near_one(options.d0.* / options.target, options.eps);
                    self.apply_origin(V, .{ .d0 = &dist, .d1 = options.d1 });
                    options.d0.* = exp2_near_zero(dist, options.eps) * options.target;

                    // Make sure our result doesn't round to 0
                    if (@typeInfo(V) != .comptime_float and !std.math.isNan(options.d0.*)) {
                        options.d0.* = @max(options.d0.*, std.math.floatEps(V));
                    }
                },
                .array => |info| for (0..info.len) |i| {
                    self.apply_exp(T, .{
                        .d0 = &options.d0[i],
                        .d1 = &options.d1[i],
                        .target = options.target[i],
                    });
                },
                .@"struct" => |info| inline for (info.field_names) |field_name| {
                    self.apply_exp(T, .{
                        .d0 = &@field(options.d0, field_name),
                        .d1 = &@field(options.d1, field_name),
                        .target = @field(options.target, field_name),
                    });
                },
                else => comptime unreachable,
            }
        }
    };
}

/// The (uniform) component float type of `V`, for defaulting `eps`. Structs must have all-matching
/// field types.
fn comp_type(T: type) type {
    switch (@typeInfo(T)) {
        .comptime_float, .float => return T,
        inline .array, .vector => |info| return info.child,
        .@"struct" => |info| {
            var F: ?type = null;
            for (info.field_types) |FieldType| {
                if (F != null) comptime assert(FieldType == F);
                F = FieldType;
            }
            return F orelse comptime_float;
        },
        else => comptime unreachable,
    }
}

test "identity is exact at dt=0" {
    const Spring = DampedSpringMat(f32);
    const spring: Spring = .init(.{ .dt = 0, .freq = 10, .ratio = 1 });
    try std.testing.expectEqual(Spring.identity(0), spring);

    var curr: f32 = 3;
    var vel: f32 = 7;
    spring.apply(f32, .{ .d0 = &curr, .d1 = &vel, .target = 100 });
    // Identity leaves position and velocity untouched (dt=0 -> d0_d1 term is 0).
    try std.testing.expectEqual(@as(f32, 3), curr);
    try std.testing.expectEqual(@as(f32, 7), vel);
}

test "inf snaps to target" {
    const Spring = DampedSpringMat(f32);
    const spring: Spring = .inf;
    var curr: f32 = 3;
    var vel: f32 = 7;
    spring.apply(f32, .{ .d0 = &curr, .d1 = &vel, .target = 100 });
    try std.testing.expectEqual(@as(f32, 100), curr);
    try std.testing.expectEqual(@as(f32, 0), vel);
}

test "nan propagates to both outputs" {
    const Spring = DampedSpringMat(f32);
    const spring: Spring = .inf;
    var curr: f32 = std.math.nan(f32);
    var vel: f32 = 2;
    spring.apply(f32, .{ .d0 = &curr, .d1 = &vel, .target = 3 });
    try std.testing.expect(std.math.isNan(curr));
    try std.testing.expect(std.math.isNan(vel));
}

test "critically damped settles toward target" {
    const Spring = DampedSpringMat(f32);
    var curr: f32 = 0;
    var vel: f32 = 0;
    const target: f32 = 10;
    // Step a critically damped spring many times; it must approach the target monotonically
    // without overshooting.
    var prev: f32 = curr;
    for (0..200) |_| {
        const spring: Spring = .init(.{ .dt = 1.0 / 60.0, .freq = 20, .ratio = 1 });
        spring.apply(f32, .{ .d0 = &curr, .d1 = &vel, .target = target });
        try std.testing.expect(curr >= prev - 1e-4); // no meaningful overshoot/backtrack
        try std.testing.expect(curr <= target + 1e-3);
        prev = curr;
    }
    try zml_testing.expect_approx_eq_abs(target, curr, 0.05);
}

test "apply on array is componentwise" {
    const Spring = DampedSpringMat(f32);
    const spring: Spring = .inf;
    var curr: [3]f32 = .{ 1, 2, 3 };
    var vel: [3]f32 = .{ 4, 5, 6 };
    spring.apply([3]f32, .{ .d0 = &curr, .d1 = &vel, .target = .{ 10, 20, 30 } });
    try std.testing.expectEqual([3]f32{ 10, 20, 30 }, curr);
    try std.testing.expectEqual([3]f32{ 0, 0, 0 }, vel);
}

test "apply_origin drives toward zero" {
    const Spring = DampedSpringMat(f32);
    const spring: Spring = .inf;
    var curr: f32 = 5;
    var vel: f32 = 9;
    spring.apply_origin(f32, .{ .d0 = &curr, .d1 = &vel });
    try std.testing.expectEqual(@as(f32, 0), curr);
    try std.testing.expectEqual(@as(f32, 0), vel);
}

test "apply_exp keeps value positive and settles" {
    const Spring = DampedSpringMat(f32);
    var scale: f32 = 0.25;
    var vel: f32 = 0;
    const target: f32 = 4;
    for (0..300) |_| {
        const spring: Spring = .init(.{ .dt = 1.0 / 60.0, .freq = 15, .ratio = 1 });
        spring.apply_exp(f32, .{ .d0 = &scale, .d1 = &vel, .target = target });
        try std.testing.expect(scale > 0);
    }
    try zml_testing.expect_approx_eq_abs(target, scale, 0.05);
}
