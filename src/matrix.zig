const std = @import("std");
const vec = @import("./root.zig").vec;

pub const Mat4f32 = Mat(f32, 4, 4);
pub const Mat4f64 = Mat(f64, 4, 4);

/// Row-major generic matrix type. `Mat(T, rows, cols)` stores `items: [rows][cols]T`, so
/// `items[r]` is row r and `items[r][c]` is the element at row r, column c. The mathematical
/// convention is column-vector (`M * v`); only the storage is row-major.
pub fn Mat(comptime T: type, comptime rows_: usize, comptime cols_: usize) type {
    return extern struct {
        const Self = @This();
        pub const rows: comptime_int = rows_;
        pub const cols: comptime_int = cols_;
        pub const Type: type = T;
        pub const is_square: bool = rows == cols;

        items: [rows][cols]T,

        /// Build from an array of rows. Row-major storage, so this is a direct copy.
        pub inline fn from_rows(values: [rows][cols]T) Self {
            return .{ .items = values };
        }

        /// Build from an array of columns; transposes into row-major storage.
        pub inline fn from_cols(values: [cols][rows]T) Self {
            var result: Self = undefined;
            inline for (0..cols) |c| {
                inline for (0..rows) |r| {
                    result.items[r][c] = values[c][r];
                }
            }
            return result;
        }

        pub fn transpose(self: Self) Mat(T, cols, rows) {
            var result: Mat(T, cols, rows) = undefined;
            for (0..cols) |i| {
                for (0..rows) |j| {
                    result.items[i][j] = self.items[j][i];
                }
            }
            return result;
        }

        /// Scalar multiplication
        pub fn scalar_mul(self: Self, scalar: T) Self {
            const flat: [rows * cols]T = @bitCast(self.items);
            var result_items: [rows * cols]T = undefined;
            for (&result_items, flat) |*result_item, item| {
                result_item.* = item * scalar;
            }
            return .{ .items = @bitCast(result_items) };
        }

        // `other` must be a matrix with as many rows as `self` has columns.
        pub fn mul(self: Self, other: anytype) Mat(T, Self.rows, @TypeOf(other).cols) {
            comptime {
                std.debug.assert(Self.cols == @TypeOf(other).rows);
                std.debug.assert(Self.Type == @TypeOf(other).Type);
            }
            // Result row i = sum_k self[i][k] * (row k of other). Rows are contiguous, so each
            // result row is one SIMD accumulation over the rows of `other`.
            const Wt = @Vector(@TypeOf(other).cols, T);
            var result: Mat(T, Self.rows, @TypeOf(other).cols) = undefined;
            for (0..Self.rows) |i| {
                result.items[i] = @as(Wt, @splat(self.items[i][0])) * @as(Wt, other.items[0]);
                for (1..Self.cols) |k| {
                    result.items[i] = @as(Wt, result.items[i]) + @as(Wt, @splat(self.items[i][k])) * @as(Wt, other.items[k]);
                }
            }
            return result;
        }

        pub fn add(self: Self, other: Self) Self {
            var result = Self{ .items = undefined };
            for (0..rows) |r| {
                for (0..cols) |c| {
                    result.items[r][c] = self.items[r][c] + other.items[r][c];
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
                result.items[index][i] = v[i];
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
                result.items[i][index] = v[i];
            }
            return result;
        }

        pub inline fn column(self: Self, index: usize) @Vector(rows, T) {
            var result: @Vector(rows, T) = undefined;
            inline for (0..rows) |r| {
                result[r] = self.items[r][index];
            }
            return result;
        }

        pub inline fn row(self: Self, index: usize) @Vector(cols, T) {
            return self.items[index];
        }

        pub fn extract(self: Self, comptime sub_row: usize, comptime sub_col: usize) Mat(T, sub_row, sub_col) {
            if (sub_col > cols or sub_row > rows) @compileError("sub matrix dimensions must be less than or equal to matrix dimensions");
            var result: Mat(T, sub_row, sub_col) = undefined;
            for (0..sub_row) |r| {
                for (0..sub_col) |c| {
                    result.items[r][c] = self.items[r][c];
                }
            }
            return result;
        }

        pub fn sub(self: Self, other: Self) Self {
            var result: Self = .{ .items = undefined };
            for (0..rows) |r| {
                for (0..cols) |c| {
                    result.items[r][c] = self.items[r][c] - other.items[r][c];
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
                result.items[3][2] = -1.0;
                result.items[2][3] = -(far * near) / (far - near);

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
                result.items[0][1] = s[1];
                result.items[0][2] = s[2];

                result.items[1][0] = u[0];
                result.items[1][1] = u[1];
                result.items[1][2] = u[2];

                result.items[2][0] = -f[0];
                result.items[2][1] = -f[1];
                result.items[2][2] = -f[2];

                result.items[0][3] = -vec.dot(s, eye);
                result.items[1][3] = -vec.dot(u, eye);
                result.items[2][3] = vec.dot(f, eye);

                return result;
            }
            unreachable;
        }

        pub fn translate(self: Self, vector: @Vector(rows - 1, T)) Self {
            if (rows == cols) {
                // The translation lives in the last column, one entry per row -- strided under
                // row-major storage.
                var result = self;
                inline for (0..rows - 1) |r| {
                    result.items[r][cols - 1] += vector[r];
                }
                return result;
            }
            unreachable;
        }

        pub inline fn position(self: Self) @Vector(rows - 1, T) {
            if (rows == cols) {
                var result: @Vector(rows - 1, T) = undefined;
                inline for (0..rows - 1) |r| {
                    result[r] = self.items[r][cols - 1];
                }
                return result;
            }
            unreachable;
        }

        /// Scaling transform matrix
        pub fn scale(self: Self, factors: @Vector(rows - 1, T)) Self {
            if (rows == cols) {
                // Post-multiply by diag(factors): scale each basis column by its factor.
                var result = self;
                inline for (0..rows - 1) |c| {
                    inline for (0..rows - 1) |r| {
                        result.items[r][c] = self.items[r][c] * factors[c];
                    }
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

                // Literal written as columns, transposed into row-major storage by `from_cols`.
                const rot: Self = .from_cols(.{
                    .{ t * a[0] * a[0] + c, t * a[0] * a[1] + s * a[2], t * a[0] * a[2] - s * a[1], 0 },
                    .{ t * a[0] * a[1] - s * a[2], t * a[1] * a[1] + c, t * a[1] * a[2] + s * a[0], 0 },
                    .{ t * a[0] * a[2] + s * a[1], t * a[1] * a[2] - s * a[0], t * a[2] * a[2] + c, 0 },
                    .{ 0, 0, 0, 1 },
                });

                return self.mul(rot);
            }
            unreachable;
        }

        /// The (rows-1)x(cols-1) sub matrix formed by deleting the given row and column.
        pub fn minor(self: Self, comptime del_row: usize, comptime del_col: usize) Mat(T, rows - 1, cols - 1) {
            var result: Mat(T, rows - 1, cols - 1) = undefined;
            var rr: usize = 0;
            inline for (0..rows) |r| {
                if (r == del_row) continue;
                var cc: usize = 0;
                inline for (0..cols) |c| {
                    if (c == del_col) continue;
                    result.items[rr][cc] = self.items[r][c];
                    cc += 1;
                }
                rr += 1;
            }
            return result;
        }

        /// The determinant of a square matrix (cofactor expansion along the first row).
        pub fn determinant(self: Self) T {
            comptime std.debug.assert(is_square);
            if (rows == 1) return self.items[0][0];
            if (rows == 2) return self.items[0][0] * self.items[1][1] - self.items[0][1] * self.items[1][0];
            var det: T = 0;
            inline for (0..cols) |c| {
                const sign: T = if (c % 2 == 0) 1 else -1;
                det += sign * self.items[0][c] * self.minor(0, c).determinant();
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
                    result.items[i][j] = sign * self.minor(j, i).determinant() * inv_det;
                }
            }
            return result;
        }

        /// Fast inverse for a rigid transform (rotation + translation, no scale/shear):
        /// transpose the 3x3 rotation and negate the rotated translation. 4x4 only.
        pub fn inverse_ortho(self: Self) Self {
            comptime std.debug.assert(rows == 4 and cols == 4);
            var result: Self = .identity;
            inline for (0..3) |i| {
                inline for (0..3) |j| {
                    result.items[i][j] = self.items[j][i];
                }
            }
            const t = self.position();
            inline for (0..3) |i| {
                var s: T = 0;
                inline for (0..3) |k| {
                    s += self.items[k][i] * t[k];
                }
                result.items[i][3] = -s;
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
            result.items[1][0] = 2 * (x * y + w * z);
            result.items[2][0] = 2 * (x * z - w * y);
            result.items[0][1] = 2 * (x * y - w * z);
            result.items[1][1] = 1 - 2 * (x * x + z * z);
            result.items[2][1] = 2 * (y * z + w * x);
            result.items[0][2] = 2 * (x * z + w * y);
            result.items[1][2] = 2 * (y * z - w * x);
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
                result.items[0][3] = -(right + left) / (right - left);
                result.items[1][3] = -(top + bottom) / (top - bottom);
                result.items[2][3] = -near / (far - near);
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
                result.items[3][2] = -1.0;
                result.items[2][3] = -near;
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
                result.items[3][2] = -1.0;
                result.items[2][3] = (far * near) / (far - near);
                return result;
            }
            unreachable;
        }

        /// Per-axis scale factors, taken from the lengths of the first three basis columns.
        pub fn get_scale(self: Self) @Vector(3, T) {
            comptime std.debug.assert(rows >= 3 and cols >= 3);
            var s: @Vector(3, T) = undefined;
            inline for (0..3) |i| {
                s[i] = vec.norm(@Vector(3, T){ self.items[0][i], self.items[1][i], self.items[2][i] });
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
                    rot.items[r][i] = self.items[r][i] * inv;
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
            const b = self.items[2][3];
            const near = b / a;
            return .{
                .near = near,
                .far = a * near / (1.0 + a),
                .fovy = 2.0 * std.math.atan(tan_half),
                .aspect = self.items[1][1] / self.items[0][0],
            };
        }

        /// Identity for square matrices, and `[I | 0]` for affine matrices whose last row is
        /// elided (`cols == rows + 1`, e.g. the 3x4 that represents a 4x4 rigid transform).
        pub const identity = blk: {
            if (rows == cols or cols == rows + 1) {
                var result: Self = .zero;
                for (0..rows) |i| {
                    result.items[i][i] = 1;
                }
                break :blk result;
            }
            @compileError("identity is only defined for square or affine (cols == rows + 1) matrices");
        };

        pub const zero: Self = .{ .items = @splat(@splat(0)) };

        pub fn format(self: @This(), writer: *std.Io.Writer) std.Io.Writer.Error!void {
            var max_widths: [cols]usize = @splat(0);
            for (0..cols) |c| {
                for (0..rows) |r| {
                    const len = std.fmt.count("{d}", .{self.items[r][c]});
                    max_widths[c] = @max(max_widths[c], len);
                }
            }

            for (0..rows) |r| {
                try writer.writeAll("[");
                for (0..cols) |c| {
                    const len = std.fmt.count("{d}", .{self.items[r][c]});
                    for (0..max_widths[c] - len) |_| {
                        try writer.writeByte(' ');
                    }
                    try writer.print("{d}", .{self.items[r][c]});
                    if (c < cols - 1) try writer.writeAll(", ");
                }
                try writer.writeByte(']');
                if (r != rows - 1) try writer.writeByte('\n');
            }
        }

        pub fn eql(self: Self, other: Self) bool {
            for (0..rows) |r| {
                for (0..cols) |c| {
                    if (self.items[r][c] != other.items[r][c]) {
                        return false;
                    }
                }
            }
            return true;
        }
    };
}

// Pins the in-memory representation of `Mat`, which every operation in this file silently
// depends on. This is deliberately the one test a change of storage convention cannot leave
// alone: any future change MUST rewrite the expected byte order below, so it shows up in review
// instead of passing quietly. Transpose-invariant checks (identity, determinant,
// `M * M.inverse()`, `from/to` round-trips) would all stay green through a wrong flip; this
// one won't.
//
// Convention: `Mat(T, rows, cols)` stores `items: [rows][cols]T` -- row-major, row count first.
// Element (row r, col c) lives at linear index `r * cols + c`.
test "storage layout is row-major" {
    const M = Mat(f32, 3, 2); // 3 rows, 2 columns
    const m: M = .{ .items = .{
        .{ 1, 2 }, // row 0: cols 0,1
        .{ 3, 4 }, // row 1
        .{ 5, 6 }, // row 2
    } };

    // Raw storage: the three rows laid end to end.
    const flat: [6]f32 = @bitCast(m.items);
    try std.testing.expectEqual([6]f32{ 1, 2, 3, 4, 5, 6 }, flat);

    // Constructors agree with that storage.
    try std.testing.expectEqual(m, M.from_rows(.{ .{ 1, 2 }, .{ 3, 4 }, .{ 5, 6 } }));
    try std.testing.expectEqual(m, M.from_cols(.{ .{ 1, 3, 5 }, .{ 2, 4, 6 } }));

    // Accessors read the logical row/column, independent of storage.
    try std.testing.expectEqual(@Vector(2, f32){ 1, 2 }, m.row(0));
    try std.testing.expectEqual(@Vector(2, f32){ 5, 6 }, m.row(2));
    try std.testing.expectEqual(@Vector(3, f32){ 1, 3, 5 }, m.column(0));
    try std.testing.expectEqual(@Vector(3, f32){ 2, 4, 6 }, m.column(1));

    // Element access is items[row][col].
    try std.testing.expectEqual(@as(f32, 4), m.items[1][1]); // row 1, col 1
    try std.testing.expectEqual(@as(f32, 6), m.items[2][1]); // row 2, col 1
}

test "format" {
    const c: Mat(f32, 3, 3) = .from_rows(.{
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
    const expected: Mat(f32, 4, 4) = .from_rows(.{
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
        const a = Mat(f32, 2, 2).from_cols(.{
            .{ 1, 2 },
            .{ 3, 4 },
        });
        const b = Mat(f32, 2, 2).from_cols(.{
            .{ 5, 6 },
            .{ 7, 8 },
        });
        const c = a.mul(b);

        const excpected_c = Mat(f32, 2, 2).from_cols(.{
            .{ 23, 34 },
            .{ 31, 46 },
        });
        try std.testing.expectEqual(excpected_c, c);
    }

    {
        // a is 3x2 (its columns are (1,2,3) and (4,5,6)); b is 2x3.
        const a = Mat(f32, 3, 2).from_cols(.{
            .{ 1, 2, 3 },
            .{ 4, 5, 6 },
        });
        const b = Mat(f32, 2, 3).from_cols(.{
            .{ 1, 2 },
            .{ 3, 4 },
            .{ 5, 6 },
        });
        const c = a.mul(b);

        const excpected_c = Mat(f32, 3, 3).from_cols(.{
            .{ 9, 12, 15 },
            .{ 19, 26, 33 },
            .{ 29, 40, 51 },
        });
        try std.testing.expectEqual(excpected_c, c);
    }

    {
        const a = Mat(f32, 4, 4).from_cols(.{
            .{ 1, 2, 3, 4 },
            .{ 5, 6, 7, 8 },
            .{ 9, 10, 11, 12 },
            .{ 13, 14, 15, 16 },
        });
        const b = Mat(f32, 4, 4).from_cols(.{
            .{ 17, 18, 19, 20 },
            .{ 21, 22, 23, 24 },
            .{ 25, 26, 27, 28 },
            .{ 29, 30, 31, 32 },
        });
        const c = a.mul(b);

        const excpected_c = Mat(f32, 4, 4).from_cols(.{
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
        const mat: Mat(f32, 4, 4) = .from_rows(.{
            .{ 1, 0, 0, 5 },
            .{ 0, 1, 0, 6 },
            .{ 0, 0, 1, 7 },
            .{ 0, 0, 0, 1 },
        });
        const scaled = mat.scale(.{ 2, 3, 4 });

        const expected: Mat(f32, 4, 4) = .from_rows(.{
            .{ 2, 0, 0, 5 },
            .{ 0, 3, 0, 6 },
            .{ 0, 0, 4, 7 },
            .{ 0, 0, 0, 1 },
        });
        try std.testing.expectEqual(expected, scaled);
    }

    {
        const mat: Mat(f32, 3, 3) = .from_rows(.{
            .{ 0.707, -0.707, 0 },
            .{ 0.707, 0.707, 0 },
            .{ 0, 0, 1 },
        });

        const scaled = mat.scale(.{ 2, 3 });

        const expected: Mat(f32, 3, 3) = .from_rows(.{
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
    // M * v: row r of the result is dot(row r of m, v). Rows are contiguous under row-major.
    var out: @Vector(4, f32) = undefined;
    inline for (0..4) |r| {
        out[r] = @reduce(.Add, @as(@Vector(4, f32), m.items[r]) * v);
    }
    return out;
}

fn expect_mat_close(comptime R: usize, comptime C: usize, expected: Mat(f32, R, C), actual: Mat(f32, R, C), tol: f32) !void {
    inline for (0..R) |r| {
        inline for (0..C) |c| {
            try std.testing.expectApproxEqAbs(expected.items[r][c], actual.items[r][c], tol);
        }
    }
}

test "determinant" {
    try std.testing.expectEqual(@as(f32, 1), Mat(f32, 4, 4).identity.determinant());

    const m2: Mat(f32, 2, 2) = .from_rows(.{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    try std.testing.expectApproxEqAbs(@as(f32, -2), m2.determinant(), 1e-6);

    const m3: Mat(f32, 3, 3) = .from_rows(.{
        .{ 2, 0, 1 },
        .{ 1, 3, 2 },
        .{ 1, 0, 1 },
    });
    try std.testing.expectApproxEqAbs(@as(f32, 3), m3.determinant(), 1e-6);
}

test "inverse" {
    const m: Mat(f32, 4, 4) = .from_rows(.{
        .{ 4, 1, 0, 0 },
        .{ 1, 3, 1, 0 },
        .{ 0, 1, 2, 1 },
        .{ 0, 0, 1, 2 },
    });
    try expect_mat_close(4, 4, .identity, m.mul(m.inverse()), 1e-4);

    const m3: Mat(f32, 3, 3) = .from_rows(.{
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
