const std = @import("std");
const vec = @import("./root.zig").vec;

pub const Mat4f32 = Mat(f32, 4, 4);
pub const Mat4f64 = Mat(f64, 4, 4);

/// column major generic matrix type
pub fn Mat(comptime T: type, comptime cols_: usize, comptime rows_: usize) type {
    return extern struct {
        const Self = @This();
        pub const rows: comptime_int = rows_;
        pub const cols: comptime_int = cols_;
        pub const Type: type = T;
        pub const is_square: bool = rows == cols;

        items: [cols][rows]T,

        pub inline fn from_column_major_array(values: [cols][rows]T) Self {
            return .{ .items = values };
        }

        /// performs a transpose operation, usefull for more human readable mat literals
        pub inline fn from_row_major_array(values: [rows][cols]T) Self {
            return Mat(T, rows, cols).from_column_major_array(values).transpose();
        }

        // generated code seams fast but tbd
        pub fn transpose(self: Self) Mat(T, rows, cols) {
            var result: Mat(T, rows, cols) = .from_column_major_array(undefined);
            for (0..cols) |c| {
                for (0..rows) |r| {
                    result.items[r][c] = self.items[c][r];
                }
            }
            return result;
        }

        /// Scalar multiplication
        pub fn scalar_mul(self: Self, scalar: T) Self {
            const items: [rows * cols]T = @bitCast(self.items);
            var result_items: [rows * cols]T = undefined;
            for (&result_items, items) |*result_item, item| {
                result_item.* = item * scalar;
            }
            return .from_column_major_array(@bitCast(result_items));
        }

        // `other` must be a matrix with the same number of rows as the numbers of columns of `self`
        pub fn mul(self: Self, other: anytype) Mat(T, @TypeOf(other).cols, Self.rows) {
            comptime {
                std.debug.assert(Self.cols == @TypeOf(other).rows);
                std.debug.assert(Self.Type == @TypeOf(other).Type);
            }
            const Wt = @Vector(rows, T);
            var result: Mat(T, @TypeOf(other).cols, Self.rows) = undefined;
            for (0..@TypeOf(result).cols) |i| {
                result.items[i] = @as(Wt, self.items[0]) * @as(Wt, @splat(other.items[i][0]));
                for (1..Self.cols) |j| {
                    result.items[i] = @as(Wt, result.items[i]) + @as(Wt, self.items[j]) * @as(Wt, @splat(other.items[i][j]));
                }
            }
            return result;
        }

        pub fn add(self: Self, other: Self) Self {
            var result = Self{ .items = undefined };
            for (0..cols) |c| {
                for (0..rows) |r| {
                    result.items[c][r] = self.items[c][r] + other.items[c][r];
                }
            }
            return result;
        }

        pub inline fn modify_row(self: Self, index: usize, v: anytype) Self {
            comptime {
                std.debug.assert(@typeInfo(@TypeOf(v)) == .vector);
                std.debug.assert(@typeInfo(@TypeOf(v)).vector.len <= cols);
            }
            var result = self;
            for (0..@typeInfo(@TypeOf(v)).vector.len) |i| {
                result.items[i][index] = v[i];
            }
            return result;
        }

        pub inline fn modify_column(self: Self, index: usize, v: anytype) Self {
            comptime {
                std.debug.assert(@typeInfo(@TypeOf(v)) == .vector);
                std.debug.assert(@typeInfo(@TypeOf(v)).vector.len <= rows);
            }

            var result = self;
            for (0..@typeInfo(@TypeOf(v)).vector.len) |i| {
                result.items[index][i] = v[i];
            }
            return result;
        }

        pub inline fn column(self: Self, index: usize) @Vector(rows, T) {
            return self.items[index];
        }

        pub inline fn row(self: Self, index: usize) @Vector(cols, T) {
            var result: @Vector(cols, T) = undefined;
            inline for (0..cols) |c| {
                result[c] = self.items[c][index];
            }
            return result;
        }

        pub fn extract(self: Self, comptime sub_col: usize, comptime sub_row: usize) Mat(T, sub_col, sub_row) {
            if (sub_col > cols or sub_row > rows) @compileError("sub matrix dimensions must be less than or equal to matrix dimensions");
            var result: Mat(T, sub_col, sub_row) = .from_column_major_array(undefined);
            for (0..sub_col) |c| {
                for (0..sub_row) |r| {
                    result.items[c][r] = self.items[c][r];
                }
            }
            return result;
        }

        pub fn sub(self: Self, other: Self) Self {
            var result: Self = .{ .items = undefined };
            for (0..cols) |c| {
                for (0..rows) |r| {
                    result.items[c][r] = self.items[c][r] - other.items[c][r];
                }
            }
            return result;
        }

        /// create a perspective projection matrix
        pub fn perspective(fovy: T, aspect: T, near: T, far: T) Self {
            if (rows == 4 and cols == 4) {
                const tanHalfFovy = std.math.tan(fovy / 2);

                var result: Self = .zero;
                result.items[0][0] = 1.0 / (aspect * tanHalfFovy);
                result.items[1][1] = 1.0 / tanHalfFovy;
                result.items[2][2] = far / (near - far);
                result.items[2][3] = -1.0;
                result.items[3][2] = -(far * near) / (far - near);

                return result;
            }
            unreachable;
        }

        /// create a look-at view matrix
        pub fn lookAt(eye: @Vector(3, T), center: @Vector(3, T), up: @Vector(3, T)) Self {
            if (rows == 4 and cols == 4) {
                const f = vec.normalize(center - eye);
                const s = vec.normalize(vec.cross(f, up));
                const u = vec.cross(s, f);

                var result: Self = .identity;
                result.items[0][0] = s[0];
                result.items[1][0] = s[1];
                result.items[2][0] = s[2];

                result.items[0][1] = u[0];
                result.items[1][1] = u[1];
                result.items[2][1] = u[2];

                result.items[0][2] = -f[0];
                result.items[1][2] = -f[1];
                result.items[2][2] = -f[2];

                result.items[3][0] = -vec.dot(s, eye);
                result.items[3][1] = -vec.dot(u, eye);
                result.items[3][2] = vec.dot(f, eye);

                return result;
            }
            unreachable;
        }

        pub fn translate(self: Self, vector: @Vector(rows - 1, T)) Self {
            if (rows == cols) {
                var result = self;
                result.items[cols - 1][0 .. rows - 1].* = self.items[cols - 1][0 .. rows - 1].* + vector;
                return result;
            }
            unreachable;
        }

        pub inline fn position(self: Self) @Vector(rows - 1, T) {
            if (rows == cols) {
                return self.items[cols - 1][0 .. rows - 1].*;
            }
            unreachable;
        }

        /// Scaling transform matrix
        pub fn scale(self: Self, factors: @Vector(rows - 1, T)) Self {
            if (rows == cols) {
                var result = self;
                inline for (0..rows - 1) |i| {
                    result.items[i][0 .. rows - 1].* = self.items[i][0 .. rows - 1].* * @as(@Vector(rows - 1, T), @splat(factors[i]));
                }
                return result;
            }
            unreachable;
        }

        pub fn rotate(self: Self, angle: T, axis: @Vector(3, T)) Self {
            if (rows == 4 and cols == 4) {
                const a = vec.normalize(axis);
                const c = std.math.cos(angle);
                const s = std.math.sin(angle);
                const t = 1.0 - c;

                const rot: Self = .{
                    .items = .{
                        .{ t * a[0] * a[0] + c, t * a[0] * a[1] + s * a[2], t * a[0] * a[2] - s * a[1], 0 },
                        .{ t * a[0] * a[1] - s * a[2], t * a[1] * a[1] + c, t * a[1] * a[2] + s * a[0], 0 },
                        .{ t * a[0] * a[2] + s * a[1], t * a[1] * a[2] - s * a[0], t * a[2] * a[2] + c, 0 },
                        .{ 0, 0, 0, 1 },
                    },
                };

                return self.mul(rot);
            }
            unreachable;
        }

        /// The (cols-1)x(rows-1) sub matrix formed by deleting the given column and row.
        pub fn minor(self: Self, comptime del_row: usize, comptime del_col: usize) Mat(T, cols - 1, rows - 1) {
            var result: Mat(T, cols - 1, rows - 1) = undefined;
            var cc: usize = 0;
            inline for (0..cols) |c| {
                if (c == del_col) continue;
                var rr: usize = 0;
                inline for (0..rows) |r| {
                    if (r == del_row) continue;
                    result.items[cc][rr] = self.items[c][r];
                    rr += 1;
                }
                cc += 1;
            }
            return result;
        }

        /// The determinant of a square matrix (cofactor expansion along the first row).
        pub fn determinant(self: Self) T {
            comptime std.debug.assert(is_square);
            if (rows == 1) return self.items[0][0];
            if (rows == 2) return self.items[0][0] * self.items[1][1] - self.items[1][0] * self.items[0][1];
            var det: T = 0;
            inline for (0..cols) |c| {
                const sign: T = if (c % 2 == 0) 1 else -1;
                det += sign * self.items[c][0] * self.minor(0, c).determinant();
            }
            return det;
        }

        /// The inverse of a square matrix via the adjugate (cofactor) method.
        /// Asserts the matrix is square; the determinant must be non-zero.
        pub fn inverse(self: Self) Self {
            comptime std.debug.assert(is_square and rows >= 2);
            const inv_det = 1.0 / self.determinant();
            var result: Self = undefined;
            // A^-1[row=i, col=j] = cofactor(j, i) / det   (adjugate transposes the cofactor matrix)
            inline for (0..rows) |i| {
                inline for (0..cols) |j| {
                    const sign: T = if ((i + j) % 2 == 0) 1 else -1;
                    result.items[j][i] = sign * self.minor(j, i).determinant() * inv_det;
                }
            }
            return result;
        }

        /// Fast inverse for a rigid transform (rotation + translation, no scale/shear):
        /// transpose the 3x3 rotation and negate the rotated translation. 4x4 only.
        pub fn inverse_ortho(self: Self) Self {
            comptime std.debug.assert(rows == 4 and cols == 4);
            var result: Self = .identity;
            inline for (0..3) |c| {
                inline for (0..3) |r| {
                    result.items[c][r] = self.items[r][c];
                }
            }
            const t = self.position();
            inline for (0..3) |r| {
                var s: T = 0;
                inline for (0..3) |k| {
                    s += self.items[r][k] * t[k];
                }
                result.items[3][r] = -s;
            }
            return result;
        }

        /// Build a rotation matrix from a quaternion (x, y, z, w). Works for 3x3 and 4x4.
        pub fn from_quat(q: @Vector(4, T)) Self {
            comptime std.debug.assert(is_square and (rows == 3 or rows == 4));
            const x = q[0];
            const y = q[1];
            const z = q[2];
            const w = q[3];
            var result: Self = .identity;
            result.items[0][0] = 1 - 2 * (y * y + z * z);
            result.items[0][1] = 2 * (x * y + w * z);
            result.items[0][2] = 2 * (x * z - w * y);
            result.items[1][0] = 2 * (x * y - w * z);
            result.items[1][1] = 1 - 2 * (x * x + z * z);
            result.items[1][2] = 2 * (y * z + w * x);
            result.items[2][0] = 2 * (x * z + w * y);
            result.items[2][1] = 2 * (y * z - w * x);
            result.items[2][2] = 1 - 2 * (x * x + y * y);
            return result;
        }

        /// Right-handed orthographic projection mapping depth to [0, 1] (matches `perspective`). 4x4 only.
        pub fn orthographic(left: T, right: T, bottom: T, top: T, near: T, far: T) Self {
            if (rows == 4 and cols == 4) {
                var result: Self = .identity;
                result.items[0][0] = 2.0 / (right - left);
                result.items[1][1] = 2.0 / (top - bottom);
                result.items[2][2] = -1.0 / (far - near);
                result.items[3][0] = -(right + left) / (right - left);
                result.items[3][1] = -(top + bottom) / (top - bottom);
                result.items[3][2] = -near / (far - near);
                return result;
            }
            unreachable;
        }

        /// Right-handed perspective projection with an infinite far plane, depth range [0, 1]. 4x4 only.
        pub fn perspective_infinite(fovy: T, aspect: T, near: T) Self {
            if (rows == 4 and cols == 4) {
                const tan_half_fovy = std.math.tan(fovy / 2);
                var result: Self = .zero;
                result.items[0][0] = 1.0 / (aspect * tan_half_fovy);
                result.items[1][1] = 1.0 / tan_half_fovy;
                result.items[2][2] = -1.0;
                result.items[2][3] = -1.0;
                result.items[3][2] = -near;
                return result;
            }
            unreachable;
        }

        /// Right-handed reversed-Z perspective (near -> 1, far -> 0), depth range [0, 1].
        /// Reversed-Z maximizes floating-point depth precision. 4x4 only.
        pub fn perspective_reverse_z(fovy: T, aspect: T, near: T, far: T) Self {
            if (rows == 4 and cols == 4) {
                const tan_half_fovy = std.math.tan(fovy / 2);
                var result: Self = .zero;
                result.items[0][0] = 1.0 / (aspect * tan_half_fovy);
                result.items[1][1] = 1.0 / tan_half_fovy;
                result.items[2][2] = near / (far - near);
                result.items[2][3] = -1.0;
                result.items[3][2] = (far * near) / (far - near);
                return result;
            }
            unreachable;
        }

        /// Per-axis scale factors, taken from the lengths of the first three basis columns.
        pub fn get_scale(self: Self) @Vector(3, T) {
            comptime std.debug.assert(rows >= 3 and cols >= 3);
            var s: @Vector(3, T) = undefined;
            inline for (0..3) |i| {
                s[i] = vec.norm(@Vector(3, T){ self.items[i][0], self.items[i][1], self.items[i][2] });
            }
            return s;
        }

        pub const Decomposed = struct {
            translation: @Vector(3, T),
            rotation: @Vector(4, T), // quaternion (x, y, z, w)
            scale: @Vector(3, T),
        };

        /// Decompose an affine transform into translation, rotation (quaternion) and scale.
        /// Assumes no shear and non-negative scale. 4x4 only.
        pub fn decompose(self: Self) Decomposed {
            comptime std.debug.assert(rows == 4 and cols == 4);
            const s = self.get_scale();
            var rot = self;
            inline for (0..3) |i| {
                const inv = 1.0 / s[i];
                inline for (0..3) |r| {
                    rot.items[i][r] = self.items[i][r] * inv;
                }
            }
            return .{
                .translation = self.position(),
                .rotation = @import("quat.zig").from_matrix(rot),
                .scale = s,
            };
        }

        pub const Projection = struct {
            near: T,
            far: T,
            fovy: T,
            aspect: T,
        };

        /// Recover the parameters of a right-handed [0, 1]-depth perspective projection
        /// (as produced by `perspective`). 4x4 only.
        pub fn decompose_projection(self: Self) Projection {
            comptime std.debug.assert(rows == 4 and cols == 4);
            const tan_half = 1.0 / self.items[1][1];
            const a = self.items[2][2];
            const b = self.items[3][2];
            const near = b / a;
            return .{
                .near = near,
                .far = a * near / (1.0 + a),
                .fovy = 2.0 * std.math.atan(tan_half),
                .aspect = self.items[1][1] / self.items[0][0],
            };
        }

        pub const identity = blk: {
            if (rows == cols) {
                var result: Self = .zero;
                for (0..cols) |i| {
                    result.items[i][i] = 1;
                }
                break :blk result;
            }
            unreachable;
        };

        pub const zero: Self = .from_column_major_array(@splat(@splat(0)));

        pub fn format(self: @This(), writer: *std.Io.Writer) std.Io.Writer.Error!void {
            var max_widths: [cols]usize = @splat(0);
            for (0..cols) |c| {
                for (0..rows) |r| {
                    const len = std.fmt.count("{d}", .{self.items[c][r]});
                    max_widths[c] = @max(max_widths[c], len);
                }
            }

            for (0..rows) |r| {
                try writer.writeAll("[");
                for (0..cols) |c| {
                    const len = std.fmt.count("{d}", .{self.items[c][r]});
                    for (0..max_widths[c] - len) |_| {
                        try writer.writeByte(' ');
                    }
                    try writer.print("{d}", .{self.items[c][r]});
                    if (c < cols - 1) try writer.writeAll(", ");
                }
                try writer.writeByte(']');
                if (r != rows - 1) try writer.writeByte('\n');
            }
        }

        pub fn eql(self: Self, other: Self) bool {
            for (0..cols) |c| {
                for (0..rows) |r| {
                    if (self.items[c][r] != other.items[c][r]) {
                        return false;
                    }
                }
            }
            return true;
        }
    };
}

test "format" {
    const c: Mat(f32, 3, 3) = .from_row_major_array(.{
        .{ 9, 12, 15 },
        .{ 19, 26, 33 },
        .{ 29, 40, 51 },
    });
    var buff: [128]u8 = undefined;
    const result = try std.fmt.bufPrint(&buff, "{f}", .{c});
    try std.testing.expectEqualStrings(
        \\[ 9, 12, 15]
        \\[19, 26, 33]
        \\[29, 40, 51]
    , result);
}

test "translate" {
    const c = Mat(f32, 4, 4).identity.translate(.{ 1, 2, 3 });
    const expected: Mat(f32, 4, 4) = .from_row_major_array(.{
        .{ 1, 0, 0, 1 },
        .{ 0, 1, 0, 2 },
        .{ 0, 0, 1, 3 },
        .{ 0, 0, 0, 1 },
    });
    try std.testing.expectEqual(expected, c);
}

test "modify_column" {
    var m = Mat(f32, 4, 4).zero;
    m = m.modify_column(0, @Vector(4, f32){ 1, 2, 3, 4 });
    try std.testing.expectEqual(m.column(0), .{ 1, 2, 3, 4 });
    m = m.modify_column(0, @Vector(3, f32){ 5, 6, 7 });
    try std.testing.expectEqual(m.column(0), .{ 5, 6, 7, 4 });

    m = m.modify_column(0, @Vector(4, f32){ 8, 9, 10, 0 });
    m = m.modify_column(1, @Vector(4, f32){ 11, 12, 13, 0 });
    m = m.modify_column(2, @Vector(4, f32){ 14, 15, 16, 0 });
    m = m.modify_column(3, @Vector(4, f32){ 17, 18, 19, 0 });

    try std.testing.expectEqual(m.column(0), .{ 8, 9, 10, 0 });
    try std.testing.expectEqual(m.column(1), .{ 11, 12, 13, 0 });
    try std.testing.expectEqual(m.column(2), .{ 14, 15, 16, 0 });
    try std.testing.expectEqual(m.position(), .{ 17, 18, 19 });
}

test "mul" {
    {
        const a = Mat(f32, 2, 2).from_column_major_array(.{
            .{ 1, 2 },
            .{ 3, 4 },
        });
        const b = Mat(f32, 2, 2).from_column_major_array(.{
            .{ 5, 6 },
            .{ 7, 8 },
        });
        const c = a.mul(b);

        const excpected_c = Mat(f32, 2, 2).from_column_major_array(.{
            .{ 23, 34 },
            .{ 31, 46 },
        });
        try std.testing.expectEqual(excpected_c, c);
    }

    {
        const a = Mat(f32, 2, 3).from_column_major_array(.{
            .{ 1, 2, 3 },
            .{ 4, 5, 6 },
        });
        const b = Mat(f32, 3, 2).from_column_major_array(.{
            .{ 1, 2 },
            .{ 3, 4 },
            .{ 5, 6 },
        });
        const c = a.mul(b);

        const excpected_c = Mat(f32, 3, 3).from_column_major_array(.{
            .{ 9, 12, 15 },
            .{ 19, 26, 33 },
            .{ 29, 40, 51 },
        });
        try std.testing.expectEqual(excpected_c, c);
    }

    {
        const a = Mat(f32, 4, 4).from_column_major_array(.{
            .{ 1, 2, 3, 4 },
            .{ 5, 6, 7, 8 },
            .{ 9, 10, 11, 12 },
            .{ 13, 14, 15, 16 },
        });
        const b = Mat(f32, 4, 4).from_column_major_array(.{
            .{ 17, 18, 19, 20 },
            .{ 21, 22, 23, 24 },
            .{ 25, 26, 27, 28 },
            .{ 29, 30, 31, 32 },
        });
        const c = a.mul(b);

        const excpected_c = Mat(f32, 4, 4).from_column_major_array(.{
            .{ 538, 612, 686, 760 },
            .{ 650, 740, 830, 920 },
            .{ 762, 868, 974, 1080 },
            .{ 874, 996, 1118, 1240 },
        });
        try std.testing.expectEqual(excpected_c, c);
    }
}

test "scale" {
    {
        const mat: Mat(f32, 4, 4) = .from_row_major_array(.{
            .{ 1, 0, 0, 5 },
            .{ 0, 1, 0, 6 },
            .{ 0, 0, 1, 7 },
            .{ 0, 0, 0, 1 },
        });
        const scaled = mat.scale(.{ 2, 3, 4 });

        const expected: Mat(f32, 4, 4) = .from_row_major_array(.{
            .{ 2, 0, 0, 5 },
            .{ 0, 3, 0, 6 },
            .{ 0, 0, 4, 7 },
            .{ 0, 0, 0, 1 },
        });
        try std.testing.expectEqual(expected, scaled);
    }

    {
        const mat: Mat(f32, 3, 3) = .from_row_major_array(.{
            .{ 0.707, -0.707, 0 },
            .{ 0.707, 0.707, 0 },
            .{ 0, 0, 1 },
        });

        const scaled = mat.scale(.{ 2, 3 });

        const expected: Mat(f32, 3, 3) = .from_row_major_array(.{
            .{ 1.414, -2.121, 0 },
            .{ 1.414, 2.121, 0 },
            .{ 0, 0, 1 },
        });

        for (0..3) |c| {
            for (0..3) |r| {
                try std.testing.expectApproxEqAbs(expected.items[c][r], scaled.items[c][r], 0.001);
            }
        }
    }
}

fn mul_vec4(m: Mat(f32, 4, 4), v: @Vector(4, f32)) @Vector(4, f32) {
    var out: @Vector(4, f32) = .{ 0, 0, 0, 0 };
    inline for (0..4) |c| {
        out += @as(@Vector(4, f32), m.items[c]) * @as(@Vector(4, f32), @splat(v[c]));
    }
    return out;
}

fn expect_mat_close(comptime C: usize, comptime R: usize, expected: Mat(f32, C, R), actual: Mat(f32, C, R), tol: f32) !void {
    inline for (0..C) |c| {
        inline for (0..R) |r| {
            try std.testing.expectApproxEqAbs(expected.items[c][r], actual.items[c][r], tol);
        }
    }
}

test "determinant" {
    try std.testing.expectEqual(@as(f32, 1), Mat(f32, 4, 4).identity.determinant());

    const m2: Mat(f32, 2, 2) = .from_row_major_array(.{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    try std.testing.expectApproxEqAbs(@as(f32, -2), m2.determinant(), 1e-6);

    const m3: Mat(f32, 3, 3) = .from_row_major_array(.{
        .{ 2, 0, 1 },
        .{ 1, 3, 2 },
        .{ 1, 0, 1 },
    });
    try std.testing.expectApproxEqAbs(@as(f32, 3), m3.determinant(), 1e-6);
}

test "inverse" {
    const m: Mat(f32, 4, 4) = .from_row_major_array(.{
        .{ 4, 1, 0, 0 },
        .{ 1, 3, 1, 0 },
        .{ 0, 1, 2, 1 },
        .{ 0, 0, 1, 2 },
    });
    try expect_mat_close(4, 4, .identity, m.mul(m.inverse()), 1e-4);

    const m3: Mat(f32, 3, 3) = .from_row_major_array(.{
        .{ 2, 0, 1 },
        .{ 1, 3, 2 },
        .{ 1, 0, 1 },
    });
    try expect_mat_close(3, 3, .identity, m3.mul(m3.inverse()), 1e-4);
}

test "inverse_ortho" {
    const quat = @import("quat.zig");
    const q = quat.from_rotation(@Vector(3, f32){ 0, 1, 0 }, 0.7);
    var m = Mat(f32, 4, 4).from_quat(q);
    m = m.translate(.{ 3, -2, 5 });

    try expect_mat_close(4, 4, .identity, m.mul(m.inverse_ortho()), 1e-5);
    // For a rigid transform inverse_ortho must agree with the general inverse.
    try expect_mat_close(4, 4, m.inverse(), m.inverse_ortho(), 1e-4);
}

test "from_quat" {
    const quat = @import("quat.zig");
    const axis = @Vector(3, f32){ 0, 0, 1 };
    const angle: f32 = std.math.pi / 3.0;
    const q = quat.from_rotation(axis, angle);
    // from_quat must match the equivalent axis-angle rotation matrix.
    try expect_mat_close(4, 4, Mat(f32, 4, 4).identity.rotate(angle, axis), Mat(f32, 4, 4).from_quat(q), 1e-5);
}

test "orthographic" {
    const m = Mat(f32, 4, 4).orthographic(-2, 2, -1, 1, 0.5, 10);
    // near corner (left, bottom, -near) -> NDC (-1, -1, 0)
    const near_corner = mul_vec4(m, .{ -2, -1, -0.5, 1 });
    try std.testing.expectApproxEqAbs(@as(f32, -1), near_corner[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1), near_corner[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0), near_corner[2], 1e-5);
    // far plane -> z_ndc 1
    const far_pt = mul_vec4(m, .{ 0, 0, -10, 1 });
    try std.testing.expectApproxEqAbs(@as(f32, 1), far_pt[2], 1e-5);
}

test "perspective_reverse_z" {
    const near: f32 = 0.1;
    const far: f32 = 100.0;
    const m = Mat(f32, 4, 4).perspective_reverse_z(std.math.pi / 2.0, 1.0, near, far);
    const at_near = mul_vec4(m, .{ 0, 0, -near, 1 });
    try std.testing.expectApproxEqAbs(@as(f32, 1), at_near[2] / at_near[3], 1e-4);
    const at_far = mul_vec4(m, .{ 0, 0, -far, 1 });
    try std.testing.expectApproxEqAbs(@as(f32, 0), at_far[2] / at_far[3], 1e-4);
}

test "perspective_infinite" {
    const near: f32 = 0.1;
    const m = Mat(f32, 4, 4).perspective_infinite(std.math.pi / 2.0, 1.0, near);
    const at_near = mul_vec4(m, .{ 0, 0, -near, 1 });
    try std.testing.expectApproxEqAbs(@as(f32, 0), at_near[2] / at_near[3], 1e-4);
    const far_away = mul_vec4(m, .{ 0, 0, -1.0e9, 1 });
    try std.testing.expectApproxEqAbs(@as(f32, 1), far_away[2] / far_away[3], 1e-3);
}

test "get_scale" {
    const s = Mat(f32, 4, 4).identity.scale(.{ 2, 3, 4 }).get_scale();
    try std.testing.expectApproxEqAbs(@as(f32, 2), s[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3), s[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4), s[2], 1e-5);
}

test "decompose" {
    const quat = @import("quat.zig");
    const q = quat.from_rotation(vec.normalize(@Vector(3, f32){ 1, 1, 0 }), 0.5);
    var m = Mat(f32, 4, 4).from_quat(q);
    m = m.scale(.{ 2, 3, 4 });
    m = m.translate(.{ 4, -1, 7 });

    const d = m.decompose();
    try std.testing.expectApproxEqAbs(@as(f32, 2), d.scale[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 3), d.scale[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 4), d.scale[2], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 4), d.translation[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 7), d.translation[2], 1e-4);

    // Recomposing must reproduce the original matrix.
    var m2 = Mat(f32, 4, 4).from_quat(d.rotation);
    m2 = m2.scale(d.scale);
    m2 = m2.translate(d.translation);
    try expect_mat_close(4, 4, m, m2, 1e-3);
}

test "decompose_projection" {
    const fovy: f32 = std.math.pi / 3.0;
    const aspect: f32 = 1.6;
    const near: f32 = 0.2;
    const far: f32 = 75.0;
    const d = Mat(f32, 4, 4).perspective(fovy, aspect, near, far).decompose_projection();
    try std.testing.expectApproxEqAbs(near, d.near, 1e-3);
    try std.testing.expectApproxEqAbs(far, d.far, 5e-2);
    try std.testing.expectApproxEqAbs(fovy, d.fovy, 1e-4);
    try std.testing.expectApproxEqAbs(aspect, d.aspect, 1e-4);
}
