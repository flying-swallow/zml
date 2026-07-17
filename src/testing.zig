const vector = @import("vector.zig");
const std = @import("std");

pub fn expect_is_close(vec: anytype, other: @TypeOf(vec), epsilon: std.meta.Child(@TypeOf(vec))) !void {
    if (vector.distance(vec, other) > @as(std.meta.Child(@TypeOf(vec)), epsilon)) {
        std.debug.print("Vectors not close enough: {} vs {}\n", .{ vec, other });
        return error.TestExpectedApproxIsClose;
    }
}

/// Like `std.testing.expectApproxEqAbs`, but recurses through structs, vectors and arrays of
/// floats (comparing element by element). Two values compare equal when both are NaN, so tests
/// of degenerate inputs stay meaningful.
pub fn expect_approx_eq_abs(expected: anytype, actual: @TypeOf(expected), tolerance: f64) !void {
    const T = @TypeOf(expected);
    switch (@typeInfo(T)) {
        .float, .comptime_float => {
            if (std.math.isNan(expected) and std.math.isNan(actual)) return;
            if (@abs(expected - actual) > tolerance) {
                std.debug.print("expected {d}, found {d} (tol {d})\n", .{ expected, actual, tolerance });
                return error.TestExpectedApproxEqAbs;
            }
        },
        .vector => |info| {
            inline for (0..info.len) |i| {
                try expect_approx_eq_abs(expected[i], actual[i], tolerance);
            }
        },
        .array => {
            for (expected, actual) |e, a| {
                try expect_approx_eq_abs(e, a, tolerance);
            }
        },
        .@"struct" => |info| {
            inline for (info.field_names) |field_name| {
                try expect_approx_eq_abs(@field(expected, field_name), @field(actual, field_name), tolerance);
            }
        },
        else => try std.testing.expectEqual(expected, actual),
    }
}
