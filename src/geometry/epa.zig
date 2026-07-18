//! Expanding Polytope Algorithm: penetration depth, contact points and penetration axis for two
//! overlapping convex objects. Seeds from a GJK simplex, then grows a convex hull of the Minkowski
//! difference (removing faces that face each new support point, rather than splitting them) until
//! the closest face to the origin is found. Ported from JoltPhysics `EPAPenetrationDepth.h` +
//! `EPAConvexHullBuilder.h` (van den Bergen).
//!
//! Objects must expose a maximizing `get_support(direction)` and a `child` scalar type — the zml
//! primitives (Sphere, OrientedBoundedBox, Capsule, Cylinder) and the `convex_support.zig` wrappers
//! all qualify. The hull is index-based and allocation-free; instead of Jolt's binary heap the
//! closest active triangle is found by a linear scan (the polytope is small).

const std = @import("std");
const zml = @import("../root.zig");
const GJK = @import("gjk.zig").GJK;

/// EPA solver for scalar type `T`. Construct with `EPA(f32){}` and reuse across queries.
pub fn EPA(comptime T: type) type {
    return struct {
        const Self = @This();
        const V = @Vector(3, T);

        const c_max_triangles = 256;
        const c_max_points = 128;
        const c_max_points_to_include_origin = 32;
        const c_max_edge_length = 128;
        const c_min_triangle_area = 1.0e-10;
        const c_barycentric_epsilon = 1.0e-3;
        const c_near_zero_sq = 1.0e-12;

        const Edge = struct {
            neighbour_triangle: i32,
            neighbour_edge: i32,
            start_idx: i32,
        };

        const Triangle = struct {
            edge: [3]Edge,
            normal: V, // length is 2x the triangle area
            centroid: V,
            closest_len_sq: T, // signed: negative when the origin is in front of the triangle
            lambda: [2]T, // barycentric coords of the closest point to the origin
            lambda_relative_to_0: bool,
            closest_point_interior: bool,
            removed: bool,
            in_queue: bool,
            next_free: i32,
        };

        /// Penetration result: `axis` points from A to B with magnitude equal to the penetration
        /// depth, `point_a` / `point_b` are the contact points, `depth` is |point_b - point_a|.
        pub const Result = struct {
            axis: V,
            point_a: V,
            point_b: V,
            depth: T,
        };

        // Support points of the Minkowski difference and their components on A and B.
        y: [c_max_points]V = undefined,
        p: [c_max_points]V = undefined,
        q: [c_max_points]V = undefined,
        num_points: u32 = 0,

        // Triangle pool with a free list.
        triangles: [c_max_triangles]Triangle = undefined,
        free_head: i32 = -1,
        high_watermark: u32 = 0,

        // Active-triangle queue (indices), scanned for the closest to the origin.
        queue: [c_max_triangles]i32 = undefined,
        queue_len: u32 = 0,

        // Triangles created by the last add_point, for the defect check.
        new_triangles: [c_max_edge_length]i32 = undefined,
        new_triangles_len: u32 = 0,

        inline fn splat(s: T) V {
            return @splat(s);
        }

        inline fn tri(self: *Self, i: i32) *Triangle {
            return &self.triangles[@intCast(i)];
        }

        /// a*b - c*d with fused multiply-adds (Kahan), matching Jolt's DifferenceOfProducts.
        inline fn dop(a: T, b: T, c: T, d: T) T {
            const cd = c * d;
            const err = @mulAdd(T, -c, d, cd);
            return @mulAdd(T, a, b, -cd) + err;
        }

        inline fn cross_precise(a: V, b: V) V {
            return .{
                dop(a[1], b[2], a[2], b[1]),
                dop(a[2], b[0], a[0], b[2]),
                dop(a[0], b[1], a[1], b[0]),
            };
        }

        fn add_support(self: *Self, a: anytype, b: anytype, direction: V) i32 {
            if (self.num_points >= c_max_points) return -1;
            const sp = a.get_support(direction);
            const sq = b.get_support(-direction);
            const idx = self.num_points;
            self.y[idx] = sp - sq;
            self.p[idx] = sp;
            self.q[idx] = sq;
            self.num_points += 1;
            return @intCast(idx);
        }

        // ---- Triangle pool ----

        fn alloc_triangle(self: *Self, idx0: i32, idx1: i32, idx2: i32) i32 {
            var idx: i32 = undefined;
            if (self.free_head >= 0) {
                idx = self.free_head;
                self.free_head = self.tri(idx).next_free;
            } else if (self.high_watermark < c_max_triangles) {
                idx = @intCast(self.high_watermark);
                self.high_watermark += 1;
            } else return -1;
            self.init_triangle(idx, idx0, idx1, idx2);
            return idx;
        }

        fn free_triangle(self: *Self, idx: i32) void {
            self.tri(idx).next_free = self.free_head;
            self.free_head = idx;
        }

        /// Compute a triangle's normal, signed closest distance to the origin and barycentric
        /// coordinates of that closest point. Uses the two shortest edges for the most accurate
        /// normal. Ported from EPAConvexHullBuilder::Triangle::Triangle.
        fn init_triangle(self: *Self, idx: i32, idx0: i32, idx1: i32, idx2: i32) void {
            const t = self.tri(idx);
            t.edge[0] = .{ .neighbour_triangle = -1, .neighbour_edge = 0, .start_idx = idx0 };
            t.edge[1] = .{ .neighbour_triangle = -1, .neighbour_edge = 0, .start_idx = idx1 };
            t.edge[2] = .{ .neighbour_triangle = -1, .neighbour_edge = 0, .start_idx = idx2 };
            t.normal = splat(0);
            t.closest_len_sq = std.math.floatMax(T);
            t.lambda = .{ 0, 0 };
            t.lambda_relative_to_0 = false;
            t.closest_point_interior = false;
            t.removed = false;
            t.in_queue = false;
            t.next_free = -1;

            const y0 = self.y[@intCast(idx0)];
            const y1 = self.y[@intCast(idx1)];
            const y2 = self.y[@intCast(idx2)];
            t.centroid = (y0 + y1 + y2) * splat(1.0 / 3.0);

            const y10 = y1 - y0;
            const y20 = y2 - y0;
            const y21 = y2 - y1;
            const y20_dot = zml.vec.dot(y20, y20);
            const y21_dot = zml.vec.dot(y21, y21);

            if (y20_dot < y21_dot) {
                t.normal = cross_precise(y10, y20);
                const nlen = zml.vec.norm_sqr(t.normal);
                if (nlen > c_min_triangle_area) {
                    const c_dot_n = zml.vec.dot(t.centroid, t.normal);
                    t.closest_len_sq = @abs(c_dot_n) * c_dot_n / nlen;
                    const y10_dot = zml.vec.norm_sqr(y10);
                    const y10_dot_y20 = zml.vec.dot(y10, y20);
                    const det = dop(y10_dot, y20_dot, y10_dot_y20, y10_dot_y20);
                    if (det > 0) {
                        const y0_dot_y10 = zml.vec.dot(y0, y10);
                        const y0_dot_y20 = zml.vec.dot(y0, y20);
                        const l0 = dop(y10_dot_y20, y0_dot_y20, y20_dot, y0_dot_y10) / det;
                        const l1 = dop(y10_dot_y20, y0_dot_y10, y10_dot, y0_dot_y20) / det;
                        t.lambda = .{ l0, l1 };
                        t.lambda_relative_to_0 = true;
                        if (l0 > -c_barycentric_epsilon and l1 > -c_barycentric_epsilon and l0 + l1 < 1.0 + c_barycentric_epsilon)
                            t.closest_point_interior = true;
                    }
                }
            } else {
                t.normal = cross_precise(y10, y21);
                const nlen = zml.vec.norm_sqr(t.normal);
                if (nlen > c_min_triangle_area) {
                    const c_dot_n = zml.vec.dot(t.centroid, t.normal);
                    t.closest_len_sq = @abs(c_dot_n) * c_dot_n / nlen;
                    const y10_dot = zml.vec.norm_sqr(y10);
                    const y10_dot_y21 = zml.vec.dot(y10, y21);
                    const det = dop(y10_dot, y21_dot, y10_dot_y21, y10_dot_y21);
                    if (det > 0) {
                        const y1_dot_y10 = zml.vec.dot(y1, y10);
                        const y1_dot_y21 = zml.vec.dot(y1, y21);
                        const l0 = dop(y21_dot, y1_dot_y10, y10_dot_y21, y1_dot_y21) / det;
                        const l1 = dop(y10_dot_y21, y1_dot_y10, y10_dot, y1_dot_y21) / det;
                        t.lambda = .{ l0, l1 };
                        t.lambda_relative_to_0 = false;
                        if (l0 > -c_barycentric_epsilon and l1 > -c_barycentric_epsilon and l0 + l1 < 1.0 + c_barycentric_epsilon)
                            t.closest_point_interior = true;
                    }
                }
            }
        }

        inline fn is_facing(self: *Self, t: i32, position: V) bool {
            return zml.vec.dot(self.tri(t).normal, position - self.tri(t).centroid) > 0;
        }

        inline fn is_facing_origin(self: *Self, t: i32) bool {
            return zml.vec.dot(self.tri(t).normal, self.tri(t).centroid) < 0;
        }

        // ---- Edge linking ----

        fn link(self: *Self, t1: i32, e1: usize, t2: i32, e2: usize) void {
            self.tri(t1).edge[e1].neighbour_triangle = t2;
            self.tri(t1).edge[e1].neighbour_edge = @intCast(e2);
            self.tri(t2).edge[e2].neighbour_triangle = t1;
            self.tri(t2).edge[e2].neighbour_edge = @intCast(e1);
        }

        fn unlink(self: *Self, t: i32) void {
            inline for (0..3) |i| {
                const nb = self.tri(t).edge[i].neighbour_triangle;
                if (nb >= 0) {
                    const nb_edge = self.tri(t).edge[i].neighbour_edge;
                    self.tri(nb).edge[@intCast(nb_edge)].neighbour_triangle = -1;
                    self.tri(t).edge[i].neighbour_triangle = -1;
                }
            }
            if (!self.tri(t).in_queue) self.free_triangle(t);
        }

        // ---- Queue (linear-scan closest) ----

        fn queue_push(self: *Self, idx: i32) void {
            self.tri(idx).in_queue = true;
            self.queue[self.queue_len] = idx;
            self.queue_len += 1;
        }

        /// Index of the active triangle with the smallest closest_len_sq, or -1 if the queue is empty.
        fn queue_closest(self: *Self) i32 {
            var best: i32 = -1;
            var best_val: T = std.math.floatMax(T);
            var k: u32 = 0;
            while (k < self.queue_len) : (k += 1) {
                const ti = self.queue[k];
                const v = self.tri(ti).closest_len_sq;
                if (best < 0 or v < best_val) {
                    best = ti;
                    best_val = v;
                }
            }
            return best;
        }

        fn queue_remove(self: *Self, idx: i32) void {
            var k: u32 = 0;
            while (k < self.queue_len) : (k += 1) {
                if (self.queue[k] == idx) {
                    self.queue[k] = self.queue[self.queue_len - 1];
                    self.queue_len -= 1;
                    return;
                }
            }
        }

        inline fn queue_has_next(self: *Self) bool {
            return self.queue_len > 0;
        }

        // ---- Hull construction ----

        fn initialize(self: *Self, idx0: i32, idx1: i32, idx2: i32) bool {
            self.free_head = -1;
            self.high_watermark = 0;
            self.queue_len = 0;
            const t1 = self.alloc_triangle(idx0, idx1, idx2);
            const t2 = self.alloc_triangle(idx0, idx2, idx1);
            if (t1 < 0 or t2 < 0) return false;
            self.link(t1, 0, t2, 2);
            self.link(t1, 1, t2, 1);
            self.link(t1, 2, t2, 0);
            self.queue_push(t1);
            self.queue_push(t2);
            return true;
        }

        /// Triangle on which `position` is furthest to the front, or -1 if none faces it.
        fn find_facing_triangle(self: *Self, position: V, out_dist_sq: *T) i32 {
            var best: i32 = -1;
            var best_dist_sq: T = 0;
            var k: u32 = 0;
            while (k < self.queue_len) : (k += 1) {
                const ti = self.queue[k];
                if (self.tri(ti).removed) continue;
                const d = zml.vec.dot(self.tri(ti).normal, position - self.tri(ti).centroid);
                if (d > 0) {
                    const dist_sq = d * d / zml.vec.norm_sqr(self.tri(ti).normal);
                    if (dist_sq > best_dist_sq) {
                        best = ti;
                        best_dist_sq = dist_sq;
                    }
                }
            }
            out_dist_sq.* = best_dist_sq;
            return best;
        }

        /// Find the horizon: starting from a triangle that faces `vertex`, flag every triangle that
        /// faces it for removal and collect the boundary edges (of the surviving triangles across
        /// the horizon) into `out_edges`. Returns false on a numerical defect (island / < 3 edges).
        fn find_edge(self: *Self, facing: i32, vertex: V, out_edges: *[c_max_edge_length]Edge, out_len: *u32) bool {
            out_len.* = 0;
            self.tri(facing).removed = true;

            const StackEntry = struct { triangle: i32, edge: i32, iter: i32 };
            var stack: [c_max_edge_length]StackEntry = undefined;
            var cur: i32 = 0;
            stack[0] = .{ .triangle = facing, .edge = 0, .iter = -1 };
            var next_expected_start_idx: i32 = -1;

            while (true) {
                const e = &stack[@intCast(cur)];
                e.iter += 1;
                if (e.iter >= 3) {
                    self.unlink(e.triangle);
                    cur -= 1;
                    if (cur < 0) break;
                } else {
                    const edge_index: usize = @intCast(@mod(e.edge + e.iter, 3));
                    const edge = self.tri(e.triangle).edge[edge_index];
                    const n = edge.neighbour_triangle;
                    if (n >= 0 and !self.tri(n).removed) {
                        if (self.is_facing(n, vertex)) {
                            self.tri(n).removed = true;
                            cur += 1;
                            if (cur >= c_max_edge_length) return false;
                            stack[@intCast(cur)] = .{ .triangle = n, .edge = edge.neighbour_edge, .iter = 0 };
                        } else {
                            // Non-facing neighbour: this is a horizon edge. Detect islands.
                            if (edge.start_idx != next_expected_start_idx and next_expected_start_idx != -1) return false;
                            next_expected_start_idx = self.tri(n).edge[@intCast(edge.neighbour_edge)].start_idx;
                            if (out_len.* >= c_max_edge_length) return false;
                            out_edges[out_len.*] = edge;
                            out_len.* += 1;
                        }
                    }
                }
            }

            return out_len.* >= 3;
        }

        /// Add point `idx` to the hull, given a triangle it faces. Removes the facing region and
        /// stitches new triangles around the horizon. Returns false on a numerical defect.
        /// New triangles with a closest point closer than `closest_dist_sq` (or in front of the
        /// origin) are pushed to the queue. Ported from EPAConvexHullBuilder::AddPoint.
        fn add_point(self: *Self, facing: i32, idx: i32, closest_dist_sq: T) bool {
            const pos = self.y[@intCast(idx)];
            var edges: [c_max_edge_length]Edge = undefined;
            var num_edges: u32 = 0;
            if (!self.find_edge(facing, pos, &edges, &num_edges)) return false;

            self.new_triangles_len = 0;
            var i: u32 = 0;
            while (i < num_edges) : (i += 1) {
                const nt = self.alloc_triangle(edges[i].start_idx, edges[(i + 1) % num_edges].start_idx, idx);
                if (nt < 0) return false;
                self.new_triangles[self.new_triangles_len] = nt;
                self.new_triangles_len += 1;
                const t = self.tri(nt);
                if ((t.closest_point_interior and t.closest_len_sq < closest_dist_sq) or t.closest_len_sq < 0)
                    self.queue_push(nt);
            }

            i = 0;
            while (i < num_edges) : (i += 1) {
                self.link(self.new_triangles[i], 0, edges[i].neighbour_triangle, @intCast(edges[i].neighbour_edge));
                self.link(self.new_triangles[i], 1, self.new_triangles[(i + 1) % num_edges], 2);
            }
            return true;
        }

        // ---- Driver ----

        /// Penetration depth between overlapping convex objects `a` and `b`. Returns null if they do
        /// not penetrate or the algorithm cannot find a reliable normal. `collision_tolerance` is the
        /// GJK tolerance; `penetration_tolerance` (>= floatEps) controls EPA accuracy.
        pub fn penetration_depth(self: *Self, a: anytype, b: anytype, collision_tolerance: T, penetration_tolerance: T) ?Result {
            // GJK step (no convex radius): detect the collision and obtain the initial simplex.
            var gjk = GJK(T){};
            var v: V = .{ 1, 0, 0 };
            var dummy_a: V = undefined;
            var dummy_b: V = undefined;
            const dist_sq = gjk.get_closest_points(a, b, collision_tolerance, 0, &v, &dummy_a, &dummy_b);
            if (dist_sq > 0) return null; // separated (no penetration with zero convex radius)

            // Seed EPA from the GJK simplex.
            self.num_points = 0;
            var i: u32 = 0;
            while (i < gjk.num_points) : (i += 1) {
                self.y[i] = gjk.y[i];
                self.p[i] = gjk.p[i];
                self.q[i] = gjk.q[i];
            }
            self.num_points = gjk.num_points;

            return self.expand(a, b, penetration_tolerance);
        }

        fn expand(self: *Self, a: anytype, b: anytype, tolerance: T) ?Result {
            // Fill the simplex up to at least 3 (ideally 4) support points.
            switch (self.num_points) {
                1 => {
                    // Single point at the origin: replace with a tetrahedron around it.
                    self.num_points = 0;
                    _ = self.add_support(a, b, .{ 0, 1, 0 });
                    _ = self.add_support(a, b, .{ -1, -1, -1 });
                    _ = self.add_support(a, b, .{ 1, -1, -1 });
                    _ = self.add_support(a, b, .{ 0, -1, 1 });
                },
                2 => {
                    // Two points: add three more by rotating a perpendicular in 120-degree steps.
                    const axis = zml.vec.normalize(self.y[1] - self.y[0]);
                    const r3 = zml.Mat(T, 4, 4).identity.rotate(2.0 * std.math.pi / 3.0, axis).extract(3, 3);
                    const dir1 = zml.vec.norm_perpendicular(axis);
                    const dir2 = r3.mul(zml.vec.to_mat(dir1)).column(0);
                    const dir3 = r3.mul(zml.vec.to_mat(dir2)).column(0);
                    _ = self.add_support(a, b, dir1);
                    _ = self.add_support(a, b, dir2);
                    _ = self.add_support(a, b, dir3);
                },
                else => {}, // 3 or 4 points: enough
            }
            if (self.num_points < 3) return null;

            // Build the initial hull from the first three points, then add the rest.
            if (!self.initialize(0, 1, 2)) return null;
            var i: u32 = 3;
            while (i < self.num_points) : (i += 1) {
                var dist_sq: T = undefined;
                const t = self.find_facing_triangle(self.y[i], &dist_sq);
                if (t >= 0) {
                    if (!self.add_point(t, @intCast(i), std.math.floatMax(T))) return null;
                }
            }

            // Loop until the origin is inside the hull.
            while (true) {
                const t = self.queue_closest();
                if (t < 0) return null;
                if (self.tri(t).removed) {
                    self.queue_remove(t);
                    if (!self.queue_has_next()) return null;
                    self.free_triangle(t);
                    continue;
                }
                if (self.tri(t).closest_len_sq >= 0) break; // origin is in the hull

                self.queue_remove(t);
                const normal = self.tri(t).normal;
                const new_index = self.add_support(a, b, normal);
                if (new_index < 0) return null;
                if (!self.is_facing(t, self.y[@intCast(new_index)]) or !self.add_point(t, new_index, std.math.floatMax(T)))
                    return null;
                self.free_triangle(t);
                if (!self.queue_has_next() or self.num_points >= c_max_points_to_include_origin) return null;
            }

            // Main loop: expand toward the closest face until convergence.
            var closest_dist_sq: T = std.math.floatMax(T);
            var last: i32 = -1;
            var flip_v_sign = false;

            while (true) {
                const t = self.queue_closest();
                if (t < 0) break;
                self.queue_remove(t);
                if (self.tri(t).removed) {
                    self.free_triangle(t);
                    continue;
                }

                const t_normal = self.tri(t).normal;
                const t_closest = self.tri(t).closest_len_sq;

                // Next triangle is no closer than the best found: we're done.
                if (t_closest >= closest_dist_sq) break;

                if (last >= 0) self.free_triangle(last);
                last = t;

                const new_index = self.add_support(a, b, t_normal);
                if (new_index < 0) break;
                const w = self.y[@intCast(new_index)];

                const dot = zml.vec.dot(t_normal, w);
                if (dot < 0) return null; // separating axis found

                const dist_sq = dot * dot / zml.vec.norm_sqr(t_normal);
                if (dist_sq - t_closest < t_closest * tolerance) break; // converged
                closest_dist_sq = @min(closest_dist_sq, dist_sq);

                if (!self.is_facing(t, w)) break; // numerical precision reached

                if (!self.add_point(t, new_index, closest_dist_sq)) break;

                // If the new triangles form defects, the origin may be on the wrong side; check and stop.
                var has_defect = false;
                var di: u32 = 0;
                while (di < self.new_triangles_len) : (di += 1) {
                    if (self.is_facing_origin(self.new_triangles[di])) {
                        has_defect = true;
                        break;
                    }
                }
                if (has_defect) {
                    const w2 = a.get_support(-t_normal) - b.get_support(t_normal);
                    const dot2 = -zml.vec.dot(t_normal, w2);
                    if (dot2 < dot) flip_v_sign = true;
                    break;
                }

                if (!self.queue_has_next() or self.num_points >= c_max_points) break;
            }

            if (last < 0) return null; // hull was a plane: no penetration

            // Closest point on the last triangle to the origin = penetration vector.
            const lt = self.tri(last);
            const nlen = zml.vec.norm_sqr(lt.normal);
            var out_v = lt.normal * splat(zml.vec.dot(lt.centroid, lt.normal) / nlen);
            if (zml.vec.norm_sqr(out_v) < c_near_zero_sq) return null;
            if (flip_v_sign) out_v = -out_v;

            // Contact points from the barycentric coordinates of the closest point.
            const idx0: usize = @intCast(lt.edge[0].start_idx);
            const idx1: usize = @intCast(lt.edge[1].start_idx);
            const idx2: usize = @intCast(lt.edge[2].start_idx);
            const l0 = splat(lt.lambda[0]);
            const l1 = splat(lt.lambda[1]);
            var point_a: V = undefined;
            var point_b: V = undefined;
            if (lt.lambda_relative_to_0) {
                point_a = self.p[idx0] + (self.p[idx1] - self.p[idx0]) * l0 + (self.p[idx2] - self.p[idx0]) * l1;
                point_b = self.q[idx0] + (self.q[idx1] - self.q[idx0]) * l0 + (self.q[idx2] - self.q[idx0]) * l1;
            } else {
                point_a = self.p[idx1] + (self.p[idx0] - self.p[idx1]) * l0 + (self.p[idx2] - self.p[idx1]) * l1;
                point_b = self.q[idx1] + (self.q[idx0] - self.q[idx1]) * l0 + (self.q[idx2] - self.q[idx1]) * l1;
            }

            return .{ .axis = out_v, .point_a = point_a, .point_b = point_b, .depth = zml.vec.norm(out_v) };
        }
    };
}

const OBB = zml.geom.OrientedBoundedBox(f32);
const Mat4 = zml.Mat(f32, 4, 4);

fn box(center: @Vector(3, f32), half: @Vector(3, f32)) OBB {
    var m = Mat4.identity;
    m.items[3] = .{ center[0], center[1], center[2], 1 };
    return OBB.from_orientation_and_half_extent(m, half);
}

test "epa boxes overlapping on x" {
    var epa = EPA(f32){};
    const a = box(.{ 0, 0, 0 }, .{ 1, 1, 1 }); // x in [-1, 1]
    const b = box(.{ 1.5, 0, 0 }, .{ 1, 1, 1 }); // x in [0.5, 2.5] -> overlap 0.5 along x
    const r = epa.penetration_depth(a, b, 1.0e-4, 1.0e-3) orelse return error.ExpectedPenetration;
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), r.depth, 1.0e-3);
    // Minimum translation is along x.
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), @abs(r.axis[0]), 1.0e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0), r.axis[1], 1.0e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0), r.axis[2], 1.0e-3);
}

test "epa boxes deeper overlap on y" {
    var epa = EPA(f32){};
    const a = box(.{ 0, 0, 0 }, .{ 2, 1, 2 }); // y in [-1, 1]
    const b = box(.{ 0, 1.25, 0 }, .{ 2, 1, 2 }); // y in [0.25, 2.25] -> overlap 0.75 along y
    const r = epa.penetration_depth(a, b, 1.0e-4, 1.0e-3) orelse return error.ExpectedPenetration;
    try std.testing.expectApproxEqAbs(@as(f32, 0.75), r.depth, 1.0e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.75), @abs(r.axis[1]), 1.0e-3);
}

test "epa returns null when separated" {
    var epa = EPA(f32){};
    const a = box(.{ 0, 0, 0 }, .{ 1, 1, 1 });
    const far = box(.{ 5, 0, 0 }, .{ 1, 1, 1 });
    try std.testing.expect(epa.penetration_depth(a, far, 1.0e-4, 1.0e-3) == null);
}

test "epa spheres via wrappers" {
    const convex = @import("convex_support.zig");
    var epa = EPA(f32){};
    // Two unit spheres (point + convex radius) whose centres are 1.5 apart: penetration 2 - 1.5 = 0.5.
    const pa_obj = convex.PointConvexSupport(f32){ .point = .{ 0, 0, 0 } };
    const pb_obj = convex.PointConvexSupport(f32){ .point = .{ 1.5, 0, 0 } };
    const s1 = convex.AddConvexRadius(convex.PointConvexSupport(f32)){ .object = pa_obj, .radius = 1 };
    const s2 = convex.AddConvexRadius(convex.PointConvexSupport(f32)){ .object = pb_obj, .radius = 1 };
    const r = epa.penetration_depth(s1, s2, 1.0e-4, 1.0e-3) orelse return error.ExpectedPenetration;
    // Curved surface approximated by a polytope: allow a looser tolerance.
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), r.depth, 5.0e-2);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), @abs(r.axis[0]), 5.0e-2);
}
