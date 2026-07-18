//! Gilbert-Johnson-Keerthi convex collision: intersection test, closest points / distance, and
//! convex ray/shape casts. Templated on any object with a (maximizing) `get_support(direction)` —
//! the zml primitives (Sphere, OrientedBoundedBox, Capsule, Cylinder) and the wrappers in
//! `convex_support.zig` all qualify. Ported from JoltPhysics `Jolt/Geometry/GJKClosestPoint.h`
//! (van den Bergen). The simplex is bounded to four points, so no allocation is needed.

const std = @import("std");
const zml = @import("../root.zig");
const cp = @import("closest_point.zig");
const convex_support = @import("convex_support.zig");

/// GJK solver for scalar type `T`. Construct with `GJK(f32){}` and reuse across queries.
pub fn GJK(comptime T: type) type {
    return struct {
        const Self = @This();
        const V = @Vector(3, T);

        y: [4]V = undefined, // support points on the Minkowski difference A - B
        p: [4]V = undefined, // support points on A
        q: [4]V = undefined, // support points on B
        num_points: u32 = 0,

        inline fn splat(s: T) V {
            return @splat(s);
        }

        /// New closest point to the origin for the current simplex. Returns false (leaving the
        /// outputs untouched) if no strictly-closer point exists, i.e. we've converged. When
        /// `last_in_feature` is true the last-added point is assumed part of the closest feature.
        fn get_closest(self: *Self, comptime last_in_feature: bool, prev_v_len_sq: T, out_v: *V, out_v_len_sq: *T, out_set: *u32) bool {
            var set: u32 = undefined;
            var v: V = undefined;
            switch (self.num_points) {
                1 => {
                    set = 0b0001;
                    v = self.y[0];
                },
                2 => {
                    const r = cp.closest_point_on_line(T, self.y[0], self.y[1]);
                    v = r.point;
                    set = r.set;
                },
                3 => {
                    const r = cp.closest_point_on_triangle(T, last_in_feature, self.y[0], self.y[1], self.y[2]);
                    v = r.point;
                    set = r.set;
                },
                4 => {
                    const r = cp.closest_point_on_tetrahedron(T, last_in_feature, self.y[0], self.y[1], self.y[2], self.y[3]);
                    v = r.point;
                    set = r.set;
                },
                else => unreachable,
            }
            const v_len_sq = zml.vec.norm_sqr(v);
            // Comparison order matters: if v_len_sq is NaN this is false and we report no progress.
            if (v_len_sq < prev_v_len_sq) {
                out_v.* = v;
                out_v_len_sq.* = v_len_sq;
                out_set.* = set;
                return true;
            }
            return false;
        }

        fn get_max_y_len_sq(self: *const Self) T {
            var m = zml.vec.norm_sqr(self.y[0]);
            var i: u32 = 1;
            while (i < self.num_points) : (i += 1) m = @max(m, zml.vec.norm_sqr(self.y[i]));
            return m;
        }

        inline fn in_set(set: u32, i: u32) bool {
            return (set >> @as(u5, @intCast(i))) & 1 != 0;
        }

        /// Keep only the simplex points selected by `set` (compacts y).
        fn update_point_set_y(self: *Self, set: u32) void {
            var n: u32 = 0;
            var i: u32 = 0;
            while (i < self.num_points) : (i += 1) {
                if (in_set(set, i)) {
                    self.y[n] = self.y[i];
                    n += 1;
                }
            }
            self.num_points = n;
        }

        /// Keep only the simplex points selected by `set` (compacts p).
        fn update_point_set_p(self: *Self, set: u32) void {
            var n: u32 = 0;
            var i: u32 = 0;
            while (i < self.num_points) : (i += 1) {
                if (in_set(set, i)) {
                    self.p[n] = self.p[i];
                    n += 1;
                }
            }
            self.num_points = n;
        }

        /// Keep only the simplex points selected by `set` (compacts p and q).
        fn update_point_set_pq(self: *Self, set: u32) void {
            var n: u32 = 0;
            var i: u32 = 0;
            while (i < self.num_points) : (i += 1) {
                if (in_set(set, i)) {
                    self.p[n] = self.p[i];
                    self.q[n] = self.q[i];
                    n += 1;
                }
            }
            self.num_points = n;
        }

        /// Keep only the simplex points selected by `set` (compacts y, p and q).
        fn update_point_set_ypq(self: *Self, set: u32) void {
            var n: u32 = 0;
            var i: u32 = 0;
            while (i < self.num_points) : (i += 1) {
                if (in_set(set, i)) {
                    self.y[n] = self.y[i];
                    self.p[n] = self.p[i];
                    self.q[n] = self.q[i];
                    n += 1;
                }
            }
            self.num_points = n;
        }

        fn calculate_point_a_and_b(self: *const Self, out_a: *V, out_b: *V) void {
            switch (self.num_points) {
                1 => {
                    out_a.* = self.p[0];
                    out_b.* = self.q[0];
                },
                2 => {
                    const bl = cp.barycentric_line(T, self.y[0], self.y[1]);
                    out_a.* = self.p[0] * splat(bl.u) + self.p[1] * splat(bl.v);
                    out_b.* = self.q[0] * splat(bl.u) + self.q[1] * splat(bl.v);
                },
                3 => {
                    const bt = cp.barycentric_triangle(T, self.y[0], self.y[1], self.y[2]);
                    out_a.* = self.p[0] * splat(bt.u) + self.p[1] * splat(bt.v) + self.p[2] * splat(bt.w);
                    out_b.* = self.q[0] * splat(bt.u) + self.q[1] * splat(bt.v) + self.q[2] * splat(bt.w);
                },
                else => {}, // 0 or 4 points: outputs are invalid
            }
        }

        /// Test whether convex objects `a` and `b` intersect. `io_v` is the initial separating-axis
        /// guess (any non-zero vector works) and, on a non-intersection, is left as a separating axis
        /// from A to B (magnitude meaningless). On intersection `io_v` is set to zero.
        pub fn intersects(self: *Self, a: anytype, b: anytype, tolerance: T, io_v: *V) bool {
            const tol_sq = tolerance * tolerance;
            const eps = std.math.floatEps(T);
            self.num_points = 0;
            var prev_v_len_sq = std.math.floatMax(T);

            while (true) {
                const support_p = a.get_support(io_v.*);
                const support_q = b.get_support(-io_v.*);
                const w = support_p - support_q;

                // Support in the opposite direction of v -> separating axis found.
                if (zml.vec.dot(io_v.*, w) < 0) return false;

                self.y[self.num_points] = w;
                self.num_points += 1;

                var v_len_sq: T = undefined;
                var set: u32 = undefined;
                if (!self.get_closest(true, prev_v_len_sq, io_v, &v_len_sq, &set)) return false;

                if (set == 0xf) {
                    io_v.* = splat(0);
                    return true; // origin inside the tetrahedron
                }
                if (v_len_sq <= tol_sq) {
                    io_v.* = splat(0);
                    return true;
                }
                if (v_len_sq <= eps * self.get_max_y_len_sq()) {
                    io_v.* = splat(0);
                    return true; // machine precision reached
                }

                // Next search direction is -v (the closest point of the simplex to the origin).
                io_v.* = -io_v.*;

                if (prev_v_len_sq - v_len_sq <= eps * prev_v_len_sq) return false; // converged, separated
                prev_v_len_sq = v_len_sq;

                self.update_point_set_y(set);
            }
        }

        /// Closest points between `a` and `b`. Returns the squared distance, or `floatMax(T)` when
        /// they are further apart than `max_dist_sq`. On a positive finite return, `io_v` is the
        /// separating axis from A to B and `out_a` / `out_b` are the closest points; on a zero return
        /// (touching / overlapping) the points are invalid. `io_v` must start non-zero.
        pub fn get_closest_points(self: *Self, a: anytype, b: anytype, tolerance: T, max_dist_sq: T, io_v: *V, out_a: *V, out_b: *V) T {
            const tol_sq = tolerance * tolerance;
            const eps = std.math.floatEps(T);
            self.num_points = 0;
            var v_len_sq = zml.vec.norm_sqr(io_v.*);
            var prev_v_len_sq = std.math.floatMax(T);

            while (true) {
                const support_p = a.get_support(io_v.*);
                const support_q = b.get_support(-io_v.*);
                const w = support_p - support_q;
                const dot = zml.vec.dot(io_v.*, w);

                // Separated by more than max_dist_sq: terminate early.
                if (dot < 0 and dot * dot > v_len_sq * max_dist_sq) return std.math.floatMax(T);

                self.y[self.num_points] = w;
                self.p[self.num_points] = support_p;
                self.q[self.num_points] = support_q;
                self.num_points += 1;

                var set: u32 = undefined;
                if (!self.get_closest(true, prev_v_len_sq, io_v, &v_len_sq, &set)) {
                    self.num_points -= 1; // undo the point we just added
                    break;
                }

                if (set == 0xf) {
                    io_v.* = splat(0);
                    v_len_sq = 0;
                    break;
                }

                self.update_point_set_ypq(set);

                if (v_len_sq <= tol_sq) {
                    io_v.* = splat(0);
                    v_len_sq = 0;
                    break;
                }
                if (v_len_sq <= eps * self.get_max_y_len_sq()) {
                    io_v.* = splat(0);
                    v_len_sq = 0;
                    break;
                }

                io_v.* = -io_v.*;

                if (prev_v_len_sq - v_len_sq <= eps * prev_v_len_sq) break; // converged
                prev_v_len_sq = v_len_sq;
            }

            self.calculate_point_a_and_b(out_a, out_b);
            return v_len_sq;
        }

        /// Cast a ray `ray_origin + lambda * ray_direction`, lambda in [0, ioLambda), against convex
        /// object `a`. On a hit returns true and writes the collision fraction to `io_lambda`.
        /// Based on van den Bergen's GJK ray cast.
        pub fn cast_ray(self: *Self, ray_origin: V, ray_direction: V, tolerance: T, a: anytype, io_lambda: *T) bool {
            const tol_sq = tolerance * tolerance;
            self.num_points = 0;

            var lambda: T = 0;
            var x = ray_origin;
            var v = x - a.get_support(splat(0));
            var v_len_sq = std.math.floatMax(T);
            var allow_restart = false;

            while (true) {
                const support_p = a.get_support(v);
                const w = x - support_p;
                const v_dot_w = zml.vec.dot(v, w);

                if (v_dot_w > 0) {
                    const v_dot_r = zml.vec.dot(v, ray_direction);
                    // Guard against a division that would overflow to infinity (float exception).
                    if (v_dot_r >= -1.0e-18) return false;
                    const delta = v_dot_w / v_dot_r;
                    const old_lambda = lambda;
                    lambda -= delta;
                    if (old_lambda == lambda) break; // cannot converge further -> treat as a hit
                    if (lambda >= io_lambda.*) return false;
                    x = ray_origin + ray_direction * splat(lambda);
                    v_len_sq = std.math.floatMax(T); // x moved, so the early-out cache is stale
                    allow_restart = true;
                }

                self.p[self.num_points] = support_p;
                self.num_points += 1;
                // Y = {x} - P shifts with x, so recompute it every iteration.
                var i: u32 = 0;
                while (i < self.num_points) : (i += 1) self.y[i] = x - self.p[i];

                var set: u32 = undefined;
                if (!self.get_closest(false, v_len_sq, &v, &v_len_sq, &set)) {
                    if (!allow_restart) break; // close enough, call it a hit
                    // Rebuild the simplex from the last point once; round-off accumulates otherwise.
                    allow_restart = false;
                    self.p[0] = support_p;
                    self.num_points = 1;
                    v = x - support_p;
                    v_len_sq = std.math.floatMax(T);
                    continue;
                } else if (set == 0xf) {
                    break; // inside the tetrahedron -> hit
                }

                self.update_point_set_p(set);

                if (v_len_sq <= tol_sq) break; // x close enough to A
            }

            io_lambda.* = lambda;
            return true;
        }

        /// Cast convex object `a`, transformed by `start` and swept along `direction` for
        /// lambda in [0, ioLambda), against convex object `b`. On a hit returns true and writes the
        /// collision fraction to `io_lambda`. (No convex radius / contact-point variant.)
        pub fn cast_shape(self: *Self, start: zml.Mat(T, 4, 4), direction: V, tolerance: T, a: anytype, b: anytype, io_lambda: *T) bool {
            const transformed_a = convex_support.TransformedConvexObject(@TypeOf(a)){ .transform = start, .object = a };
            const difference = convex_support.MinkowskiDifference(@TypeOf(b), @TypeOf(transformed_a)){ .object_a = b, .object_b = transformed_a };
            return self.cast_ray(splat(0), direction, tolerance, difference, io_lambda);
        }
    };
}

const OBB = zml.geom.OrientedBoundedBox(f32);
const Mat4 = zml.Mat(f32, 4, 4);

fn box(center: @Vector(3, f32), half: @Vector(3, f32)) OBB {
    // Axis-aligned box: identity rotation, translation = center (set column 3 directly).
    var m = Mat4.identity;
    m.items[3] = .{ center[0], center[1], center[2], 1 };
    return OBB.from_orientation_and_half_extent(m, half);
}

test "gjk intersects boxes" {
    var gjk = GJK(f32){};
    const a = box(.{ 0, 0, 0 }, .{ 1, 1, 1 });
    const near = box(.{ 1.5, 0, 0 }, .{ 1, 1, 1 }); // gap 1.5 < 2 -> overlap
    const far = box(.{ 5, 0, 0 }, .{ 1, 1, 1 }); // gap 5 > 2 -> separated

    var v: @Vector(3, f32) = .{ 1, 0, 0 };
    try std.testing.expect(gjk.intersects(a, near, 1.0e-4, &v));
    v = .{ 1, 0, 0 };
    try std.testing.expect(!gjk.intersects(a, far, 1.0e-4, &v));
}

test "gjk closest points boxes" {
    var gjk = GJK(f32){};
    const a = box(.{ 0, 0, 0 }, .{ 1, 1, 1 }); // x in [-1, 1]
    const b = box(.{ 5, 0, 0 }, .{ 1, 1, 1 }); // x in [4, 6]

    var v: @Vector(3, f32) = .{ 1, 0, 0 };
    var pa: @Vector(3, f32) = undefined;
    var pb: @Vector(3, f32) = undefined;
    const dist_sq = gjk.get_closest_points(a, b, 1.0e-4, 1.0e12, &v, &pa, &pb);
    // Faces at x = 1 and x = 4 -> gap 3, squared 9.
    try std.testing.expectApproxEqAbs(@as(f32, 9), dist_sq, 1.0e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 1), pa[0], 1.0e-3); // closest point on A is on its x=1 face
    try std.testing.expectApproxEqAbs(@as(f32, 4), pb[0], 1.0e-3); // closest point on B is on its x=4 face
}

test "gjk closest points spheres via wrappers" {
    const convex = @import("convex_support.zig");
    var gjk = GJK(f32){};
    // Spheres as point + convex radius (how Jolt represents them), avoiding a normalize-at-zero.
    const pa_obj = convex.PointConvexSupport(f32){ .point = .{ 0, 0, 0 } };
    const pb_obj = convex.PointConvexSupport(f32){ .point = .{ 5, 0, 0 } };
    const s1 = convex.AddConvexRadius(convex.PointConvexSupport(f32)){ .object = pa_obj, .radius = 1 };
    const s2 = convex.AddConvexRadius(convex.PointConvexSupport(f32)){ .object = pb_obj, .radius = 1 };

    var v: @Vector(3, f32) = .{ 1, 0, 0 };
    var pa: @Vector(3, f32) = undefined;
    var pb: @Vector(3, f32) = undefined;
    const dist_sq = gjk.get_closest_points(s1, s2, 1.0e-4, 1.0e12, &v, &pa, &pb);
    // Centres 5 apart, radii 1 each -> surface gap 3, squared 9.
    try std.testing.expectApproxEqAbs(@as(f32, 9), dist_sq, 1.0e-2);
}

test "gjk cast_ray box" {
    var gjk = GJK(f32){};
    const b = box(.{ 5, 0, 0 }, .{ 1, 1, 1 }); // near face at x = 4
    var lambda: f32 = 1.0e6;
    const hit = gjk.cast_ray(.{ 0, 0, 0 }, .{ 1, 0, 0 }, 1.0e-4, b, &lambda);
    try std.testing.expect(hit);
    try std.testing.expectApproxEqAbs(@as(f32, 4), lambda, 1.0e-3);

    // A ray pointing away from the box does not hit within [0, ioLambda).
    var miss_lambda: f32 = 1.0e6;
    try std.testing.expect(!gjk.cast_ray(.{ 0, 0, 0 }, .{ -1, 0, 0 }, 1.0e-4, b, &miss_lambda));
}
