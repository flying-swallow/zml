const std = @import("std");

/// Element/scalar type of a vector or array type.
pub fn Child(comptime T: type) type {
    return switch (@typeInfo(T)) {
        .vector, .array => std.meta.Child(T),
        else => @compileError("Expected vector or array type, got: " ++ @typeName(T)),
    };
}

/// Number of elements in a vector or array type.
pub fn lengthOf(comptime T: type) comptime_int {
    return switch (@typeInfo(T)) {
        .vector => |v| v.len,
        .array => |a| a.len,
        else => @compileError("Expected vector or array type, got: " ++ @typeName(T)),
    };
}

/// The `@Vector` type equivalent to a vector-or-array type. Used to coerce inputs
/// into a SIMD vector for the actual math.
pub fn AsVector(comptime T: type) type {
    return @Vector(lengthOf(T), Child(T));
}

/// A container of the SAME kind (array vs vector) as `T`, but with `len` elements.
/// Used so a function's return shape matches its input shape.
pub fn Reshape(comptime T: type, comptime len: usize) type {
    return switch (@typeInfo(T)) {
        .vector => @Vector(len, Child(T)),
        .array => [len]Child(T),
        else => @compileError("Expected vector or array type, got: " ++ @typeName(T)),
    };
}

test Child {
    try std.testing.expect(Child(@Vector(3, f32)) == f32);
    try std.testing.expect(Child([3]f32) == f32);
    try std.testing.expect(Child([4]i32) == i32);
}

test lengthOf {
    try std.testing.expectEqual(3, lengthOf(@Vector(3, f32)));
    try std.testing.expectEqual(4, lengthOf([4]f64));
}

test AsVector {
    try std.testing.expect(AsVector(@Vector(3, f32)) == @Vector(3, f32));
    try std.testing.expect(AsVector([3]f32) == @Vector(3, f32));
    // An array coerces to its equivalent @Vector.
    const arr = [3]f32{ 1, 2, 3 };
    const v: AsVector(@TypeOf(arr)) = arr;
    try std.testing.expectEqual(@as(f32, 6), @reduce(.Add, v));
}

test Reshape {
    // Same kind is preserved, length can change.
    try std.testing.expect(Reshape(@Vector(4, f32), 2) == @Vector(2, f32));
    try std.testing.expect(Reshape([4]f32, 2) == [2]f32);
    try std.testing.expect(Reshape([3]i32, 3) == [3]i32);
    // A @Vector coerces back to its equivalent array.
    const v = @Vector(3, f32){ 1, 2, 3 };
    const arr: Reshape([3]f32, 3) = v;
    try std.testing.expectEqual(@as(f32, 2), arr[1]);
}
