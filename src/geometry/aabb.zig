const std = @import("std");
const zml = @import("../root.zig");
const geometry = zml.geom;

pub fn AABB(comptime T: type) type {
    return struct {
        pub const child: type = T;
        pub const primative_type: geometry.Primative = .AABB;
        const Self = @This();

        min: @Vector(3, T),
        max: @Vector(3, T),

        pub const empty = Self {
            .min = .{0,0,0},
            .max = .{0,0,0},
        };

        /// Get the closest point on or in this box to point
        pub fn get_closest_point(self: Self, point: @Vector(3, T)) @Vector(3, T) {
            return @min(@max(point, self.min), self.max);
        }

        pub fn get_sqr_distance_to(self: Self, point: @Vector(3, T)) T {
            return zml.vec.norm_sqr(self.get_closest_point(point) - point);
        }

        pub fn from_two_points(p1: @Vector(3, T), p2: @Vector(3, T)) @This() {
            std.debug.assert(@reduce(.And, p1 <= p2));
            return @This(){
                .min = @min(p1, p2),
                .max = @max(p1, p2),
            };
        }

        pub fn translate(self: Self, translation: @Vector(3, T)) Self {
            std.debug.assert(@reduce(.And, self.min <= self.max));
            return .{
                .min = self.min + translation,
                .max = self.max + translation,
            };
        }

        pub fn transform(self: Self, mat: zml.Mat(T, 4, 4)) Self {
            const pos = mat.position();
            var new_min = pos;
            var new_max = pos;

            inline for (0..3) |c| {
                const m_col = mat.column(c);
                const col: @Vector(3, T) = .{ m_col[0], m_col[1], m_col[2] };
                const a = col * @as(@Vector(3, T), @splat(self.min[c]));
                const b = col * @as(@Vector(3, T), @splat(self.max[c]));
                new_min += @min(a, b);
                new_max += @max(a, b);
            }
            return .{ .min = new_min, .max = new_max };
        }

        pub fn get_center(self: @This()) @Vector(3, T) {
            std.debug.assert(@reduce(.And, self.min <= self.max));
            return (self.min + self.max) * @as(@Vector(3, T), @splat(0.5));
        }

        pub fn get_size(self: @This()) @Vector(3, T) {
            std.debug.assert(@reduce(.And, self.min <= self.max));
            return self.max - self.min;
        }

        // Calculate the support vector for this convex shape
        pub fn get_support(self: @This(), direction: @Vector(3, T)) @Vector(3, T) {
            std.debug.assert(@reduce(.And, self.min <= self.max));
            return @select(T, direction < @as(@Vector(3, T), @splat(0)), self.max, self.min);
        }

        pub fn encapsulate_aabb(self: Self, a: Self) Self {
            std.debug.assert(@reduce(.And, self.min <= self.max));
            return .{ .min = @min(self.min, a.min), .max = @max(self.max, a.max) };
        }

        pub fn intersect(self: Self, inRHS: Self) Self {
            std.debug.assert(@reduce(.And, self.min <= self.max));
            return Self{
                .min = @max(self.min, inRHS.min),
                .max = @min(self.max, inRHS.max),
            };
        }

        pub fn expand_by(self: *Self, in: @Vector(3, T)) void {
            std.debug.assert(@reduce(.And, self.min <= self.max));
            self.min -= in;
            self.max += in;
        }

        pub fn surface_area(self: Self) T {
            std.debug.assert(@reduce(.And, self.min <= self.max));
            const extent = self.max - self.min;
            return 2 * zml.vec.dot(zml.vec.swizzle(extent, "yzz"), zml.vec.swizzle(extent, "xxy"));
        }

        pub fn volume(self: Self) T {
            std.debug.assert(@reduce(.And, self.min <= self.max));
            const extent = self.max - self.min;
            return @reduce(.Mul, extent);
        }

        pub fn eql(self: Self, other: Self) bool {
            return @reduce(.And, self.min == other.min) and @reduce(.And, self.max == other.max);
        }

        pub fn neql(self: Self, other: Self) bool {
            return !self.eql(other);
        }

        /// True if the box is well-formed (min <= max on every axis).
        pub fn is_valid(self: Self) bool {
            return @reduce(.And, self.min <= self.max);
        }

        /// Half of the size of the box.
        pub fn get_extent(self: Self) @Vector(3, T) {
            std.debug.assert(@reduce(.And, self.min <= self.max));
            return (self.max - self.min) * @as(@Vector(3, T), @splat(0.5));
        }

        /// A box large enough to contain anything, while keeping get_size() finite.
        pub fn biggest() Self {
            const half_max = 0.5 * std.math.floatMax(T);
            return .{ .min = @splat(-half_max), .max = @splat(half_max) };
        }

        /// True if this box fully contains `other`.
        pub fn contains(self: Self, other: Self) bool {
            std.debug.assert(@reduce(.And, self.min <= self.max));
            return @reduce(.And, self.min <= other.min) and @reduce(.And, self.max >= other.max);
        }

        /// Scale the box about the origin; handles non-uniform and negative scale.
        pub fn scaled(self: Self, scale: @Vector(3, T)) Self {
            const a = self.min * scale;
            const b = self.max * scale;
            return .{ .min = @min(a, b), .max = @max(a, b) };
        }

        /// Grow the box to include `point`.
        pub fn encapsulate(self: Self, point: @Vector(3, T)) Self {
            return .{ .min = @min(self.min, point), .max = @max(self.max, point) };
        }

        /// Grow the box to include a triangle.
        pub fn encapsulate_triangle(self: Self, v0: @Vector(3, T), v1: @Vector(3, T), v2: @Vector(3, T)) Self {
            return self.encapsulate(v0).encapsulate(v1).encapsulate(v2);
        }
    };
}

test "aabb_transform" {
    const AABBf32 = AABB(f32);
    const aabb = AABBf32.from_two_points(.{ 0.0, 0.0, 0.0 }, .{ 1.0, 1.0, 1.0 });
    var translation: zml.Mat(f32, 4, 4) = .identity;
    translation = translation.translate(.{ 1.0, 2.0, 3.0 });
    const transformed = aabb.transform(translation);
    try std.testing.expect(zml.vec.is_close_default(transformed.min, .{ 1.0, 2.0, 3.0 }));
    try std.testing.expect(zml.vec.is_close_default(transformed.max, .{ 2.0, 3.0, 4.0 }));
}

test "surface_area" {
    const AABBf32 = AABB(f32);
    const aabb = AABBf32.from_two_points(.{ 0.0, 0.0, 0.0 }, .{ 1.0, 2.0, 3.0 });
    try std.testing.expectApproxEqRel(aabb.surface_area(), 22.0, 1.0e-6);
}

test "volumn" {
    const AABBf32 = AABB(f32);
    const aabb = AABBf32.from_two_points(.{ 0.0, 0.0, 0.0 }, .{ 1.0, 2.0, 3.0 });
    try std.testing.expectApproxEqRel(aabb.volume(), 6.0, 1.0e-6);
}

test "aabb_get_support" {
    const aabb: AABB(f32) = .from_two_points(.{ 0.0, 0.0, 0.0 }, .{ 1.0, 1.0, 1.0 });

    try std.testing.expect(zml.vec.is_close_default(aabb.get_support(.{ 0.5774, 0.5774, 0.5774 }), .{ 0.0, 0.0, 0.0 }));
    try std.testing.expect(zml.vec.is_close_default(aabb.get_support(.{ 0.5774, 0.5774, -0.5774 }), .{ 0.0, 0.0, 1.0 }));
    try std.testing.expect(zml.vec.is_close_default(aabb.get_support(.{ 0.5774, -0.5774, 0.5774 }), .{ 0.0, 1.0, 0.0 }));
    try std.testing.expect(zml.vec.is_close_default(aabb.get_support(.{ 0.5774, -0.5774, -0.5774 }), .{ 0.0, 1.0, 1.0 }));
    try std.testing.expect(zml.vec.is_close_default(aabb.get_support(.{ -0.5774, 0.5774, 0.5774 }), .{ 1.0, 0.0, 0.0 }));
    try std.testing.expect(zml.vec.is_close_default(aabb.get_support(.{ -0.5774, 0.5774, -0.5774 }), .{ 1.0, 0.0, 1.0 }));
    try std.testing.expect(zml.vec.is_close_default(aabb.get_support(.{ -0.5774, -0.5774, 0.5774 }), .{ 1.0, 1.0, 0.0 }));
    try std.testing.expect(zml.vec.is_close_default(aabb.get_support(.{ -0.5774, -0.5774, -0.5774 }), .{ 1.0, 1.0, 1.0 }));
}

test "aabb eql / is_valid / contains" {
    const a: AABB(f32) = .from_two_points(.{ 0, 0, 0 }, .{ 2, 2, 2 });
    const b: AABB(f32) = .from_two_points(.{ 0, 0, 0 }, .{ 2, 2, 2 });
    const c: AABB(f32) = .from_two_points(.{ 0, 0, 0 }, .{ 1, 1, 1 });
    try std.testing.expect(a.eql(b));
    try std.testing.expect(a.neql(c));
    try std.testing.expect(a.is_valid());
    try std.testing.expect(a.contains(c)); // a fully contains the smaller c
    try std.testing.expect(!c.contains(a));
}

test "aabb get_extent / scaled" {
    const a: AABB(f32) = .from_two_points(.{ 0, 0, 0 }, .{ 2, 4, 6 });
    try std.testing.expect(zml.vec.is_close_default(a.get_extent(), .{ 1, 2, 3 }));

    const s = a.scaled(.{ 2, 2, 2 });
    try std.testing.expect(zml.vec.is_close_default(s.min, .{ 0, 0, 0 }));
    try std.testing.expect(zml.vec.is_close_default(s.max, .{ 4, 8, 12 }));

    // Negative scale flips an axis but keeps min <= max.
    const n = a.scaled(.{ -1, 1, 1 });
    try std.testing.expect(n.is_valid());
    try std.testing.expect(zml.vec.is_close_default(n.min, .{ -2, 0, 0 }));
    try std.testing.expect(zml.vec.is_close_default(n.max, .{ 0, 4, 6 }));
}

test "aabb encapsulate" {
    const a: AABB(f32) = .from_two_points(.{ 0, 0, 0 }, .{ 1, 1, 1 });
    const grown = a.encapsulate(.{ 2, 0.5, 0.5 });
    try std.testing.expect(zml.vec.is_close_default(grown.min, .{ 0, 0, 0 }));
    try std.testing.expect(zml.vec.is_close_default(grown.max, .{ 2, 1, 1 }));

    const tri = a.encapsulate_triangle(.{ -1, 0, 0 }, .{ 0, 3, 0 }, .{ 0, 0, 5 });
    try std.testing.expect(zml.vec.is_close_default(tri.min, .{ -1, 0, 0 }));
    try std.testing.expect(zml.vec.is_close_default(tri.max, .{ 1, 3, 5 }));
}

test "aabb biggest" {
    const b = AABB(f32).biggest();
    try std.testing.expect(b.is_valid());
    try std.testing.expect(std.math.isFinite(b.get_size()[0]));
}
