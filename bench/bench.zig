const std = @import("std");
const zbench = @import("zbench");
const caliper = @import("caliper");

fn benchmark_multiply(comptime size: usize) type {
    return struct {
        const Matrix512 = caliper.Mat(f32, size, size);
        a: Matrix512,
        b: Matrix512,

        fn init() @This() {
            var input: Matrix512 = undefined;
            for (&input.items, 0..) |*row, i| {
                for (row, 0..) |*col, j| {
                    col.* = @floatFromInt(i * size + j);
                }
            }
            return .{
                .a = input,
                .b = input,
            };
        }

        pub fn run(self: *@This(), _: std.mem.Allocator) void {
            std.mem.doNotOptimizeAway(@call(.never_inline, Matrix512.mul, .{ self.a, self.b }));
        }
    };
}

fn bench_sin_cos_fused(comptime size: usize) type {
    return struct {
        angles: @Vector(size, f32),

        fn init() @This() {
            var val: [size]f32 = undefined;
            for (0..size) |k| {
                val[k] = @as(f32, @floatFromInt(k)) * 0.01;
            }
            return .{ .angles = val };
        }

        pub fn run(self: *@This(), _: std.mem.Allocator) void {
           std.mem.doNotOptimizeAway(@call(.never_inline, caliper.vec.sin_cos, .{self.angles}));
        }
    };
}

fn benchmark_sin_cos_system(comptime size: usize) type {
    return struct {
        angles: [size]f32,

        fn init() @This() {
            var val: [size]f32 = undefined;
            for (0..size) |k| {
                val[k] = @as(f32, @floatFromInt(k)) * 0.01;
            }
            return .{ .angles = val };
        }

        pub fn run(self: *@This(), _: std.mem.Allocator) void {
            var sin_val: [size]f32 = undefined;
            var cos_val: [size]f32 = undefined;
            for(self.angles, 0..) |angle, i| {
                sin_val[i] = std.math.sin(angle);
                cos_val[i] = std.math.cos(angle);
            }
            std.mem.doNotOptimizeAway(sin_val);
            std.mem.doNotOptimizeAway(cos_val);
        }
    };
}

// Fill a length-`size` array with distinct positive values (kept away from zero
// so reductions/normalize stay well-conditioned).
fn ramp(comptime size: usize) [size]f32 {
    var val: [size]f32 = undefined;
    for (0..size) |k| val[k] = @as(f32, @floatFromInt(k % 251)) * 0.03 + 1.0;
    return val;
}

// `sin_cos` on a plain array: processed in native-width SIMD chunks (simd.zig).
fn bench_sin_cos_array(comptime size: usize) type {
    return struct {
        angles: [size]f32,
        fn init() @This() {
            return .{ .angles = ramp(size) };
        }
        pub fn run(self: *@This(), _: std.mem.Allocator) void {
            std.mem.doNotOptimizeAway(@call(.never_inline, caliper.vec.sin_cos, .{self.angles}));
        }
    };
}

// The remaining factories each come in an `_array` (chunked) and a `_vector`
// (single wide @Vector op) flavor over the SAME data, so a run shows the
// chunked path's cost against the wide-vector path it replaces for arrays.

fn bench_norm_array(comptime size: usize) type {
    return struct {
        data: [size]f32,
        fn init() @This() {
            return .{ .data = ramp(size) };
        }
        pub fn run(self: *@This(), _: std.mem.Allocator) void {
            std.mem.doNotOptimizeAway(@call(.never_inline, caliper.vec.norm_adv, .{ self.data, 32 }));
        }
    };
}

fn bench_norm_vector(comptime size: usize) type {
    return struct {
        data: @Vector(size, f32),
        fn init() @This() {
            return .{ .data = ramp(size) };
        }
        pub fn run(self: *@This(), _: std.mem.Allocator) void {
            std.mem.doNotOptimizeAway(@call(.never_inline, caliper.vec.norm_adv, .{ self.data, 32 }));
        }
    };
}

fn bench_dot_array(comptime size: usize) type {
    return struct {
        a: [size]f32,
        b: [size]f32,
        fn init() @This() {
            return .{ .a = ramp(size), .b = ramp(size) };
        }
        pub fn run(self: *@This(), _: std.mem.Allocator) void {
            std.mem.doNotOptimizeAway(@call(.never_inline, caliper.vec.dot, .{ self.a, self.b }));
        }
    };
}

fn bench_dot_vector(comptime size: usize) type {
    return struct {
        a: @Vector(size, f32),
        b: @Vector(size, f32),
        fn init() @This() {
            return .{ .a = ramp(size), .b = ramp(size) };
        }
        pub fn run(self: *@This(), _: std.mem.Allocator) void {
            std.mem.doNotOptimizeAway(@call(.never_inline, caliper.vec.dot, .{ self.a, self.b }));
        }
    };
}

fn bench_normalize_array(comptime size: usize) type {
    return struct {
        data: [size]f32,
        fn init() @This() {
            return .{ .data = ramp(size) };
        }
        pub fn run(self: *@This(), _: std.mem.Allocator) void {
            std.mem.doNotOptimizeAway(@call(.never_inline, caliper.vec.normalize, .{self.data}));
        }
    };
}

fn bench_normalize_vector(comptime size: usize) type {
    return struct {
        data: @Vector(size, f32),
        fn init() @This() {
            return .{ .data = ramp(size) };
        }
        pub fn run(self: *@This(), _: std.mem.Allocator) void {
            std.mem.doNotOptimizeAway(@call(.never_inline, caliper.vec.normalize, .{self.data}));
        }
    };
}

// ── shared batching harness ──────────────────────────────────────────────────
//
// Everything above works because one call already chews through 245-512 elements. The rest of the
// library is fixed-size ops instead -- a `quat.mul` is a handful of flops on a @Vector(4, f32) --
// so a single call is far too fast to time. Each benchmark below therefore runs its function over
// a batch of `N` inputs built once in `init()`.
//
// `N` is picked to keep the input array inside L1 (~32KB), otherwise the numbers measure memory
// bandwidth rather than the op: 1024 for elements up to ~32 bytes, 256 for the Mat4/OBB/SAT-sized
// ones.
//
// NOTE: zbench's `addParam` demands a *const* pointer and then @ptrCasts it to the `*Self` that
// `run` takes (see zbench.zig), and `&Factory(N).init()` may well be folded into .rodata -- so
// `run` must never mutate `self`. Anything mutable (the GJK solver) is a local inside `run`.

/// Times `f` over a batch of `N` pre-built argument tuples; `gen(i)` builds element `i`.
fn batch(comptime N: usize, comptime gen: anytype, comptime f: anytype) type {
    const Args = @typeInfo(@TypeOf(gen)).@"fn".return_type.?;
    return struct {
        inputs: [N]Args,
        fn init() @This() {
            var in: [N]Args = undefined;
            for (0..N) |i| in[i] = gen(i);
            return .{ .inputs = in };
        }
        pub fn run(self: *@This(), _: std.mem.Allocator) void {
            for (self.inputs) |a| std.mem.doNotOptimizeAway(@call(.never_inline, f, a));
        }
    };
}

const Vec3 = @Vector(3, f32);
const Vec4 = @Vector(4, f32);
const Mat4 = caliper.Mat4f32;
const Affine3 = caliper.Mat3x4f32;
const AABBf = caliper.geom.AABB(f32);
const Spheref = caliper.geom.Sphere(f32);
const Capsulef = caliper.geom.Capsule(f32);
const Cylinderf = caliper.geom.Cylinder(f32);
const OBBf = caliper.geom.OrientedBoundedBox(f32);
const Planef = caliper.geom.Plane(f32);
const InvDirf = caliper.geom.ray.InvDirection(f32);

// ── deterministic inputs ─────────────────────────────────────────────────────

/// Pseudo-random f32 in [-1, 1) from an element index and a `stream` id (so several independent
/// values can be drawn per element). An integer hash rather than std.Random, for the same reason
/// `ramp` is what it is: reproducible run to run, and comptime-evaluable so `init()` can fold.
fn rnd(i: usize, stream: u32) f32 {
    var x: u32 = @truncate(i);
    x = (x *% 0x9E3779B1) ^ (stream *% 0x85EBCA6B);
    x ^= x >> 16;
    x *%= 0x7FEB352D;
    x ^= x >> 15;
    x *%= 0x846CA68B;
    x ^= x >> 16;
    // Top 24 bits -> [0, 2) -> [-1, 1).
    return @as(f32, @floatFromInt(x >> 8)) * (2.0 / 16777216.0) - 1.0;
}

fn rnd_vec3(i: usize, stream: u32) Vec3 {
    return .{ rnd(i, stream), rnd(i, stream +% 1), rnd(i, stream +% 2) };
}

/// Unit-length direction, falling back to +x for a degenerate near-zero draw.
fn rnd_unit_vec3(i: usize, stream: u32) Vec3 {
    const v = rnd_vec3(i, stream);
    const n = caliper.vec.norm(v);
    if (n < 1.0e-4) return .{ 1, 0, 0 };
    return v / @as(Vec3, @splat(n));
}

/// A *unit* quaternion -- `slerp` and `rotate_vector` both assume normalized input.
fn rnd_quat(i: usize, stream: u32) Vec4 {
    return caliper.quat.from_rotation(rnd_unit_vec3(i, stream), rnd(i, stream +% 3) * std.math.pi);
}

/// A rigid (rotation + translation) transform: always invertible, and orthonormal so the same
/// matrices are also legal input for `inverse_ortho`.
fn rnd_rigid_mat4(i: usize, stream: u32) Mat4 {
    return Mat4.identity
        .rotate(rnd(i, stream) * std.math.pi, rnd_unit_vec3(i, stream +% 1))
        .translate(rnd_vec3(i, stream +% 5) * @as(Vec3, @splat(3)));
}

fn rnd_aabb(i: usize, stream: u32) AABBf {
    const c = rnd_vec3(i, stream) * @as(Vec3, @splat(4));
    // from_two_points asserts p1 <= p2, so build from a centre and a strictly positive half size.
    const h = @abs(rnd_vec3(i, stream +% 3)) + @as(Vec3, @splat(0.25));
    return .from_two_points(c - h, c + h);
}

fn rnd_sphere(i: usize, stream: u32) Spheref {
    return .from_center_radius(rnd_vec3(i, stream) * @as(Vec3, @splat(4)), @abs(rnd(i, stream +% 3)) + 0.25);
}

fn rnd_capsule(i: usize, stream: u32) Capsulef {
    const c = rnd_vec3(i, stream) * @as(Vec3, @splat(4));
    const half_axis = rnd_unit_vec3(i, stream +% 3);
    return .{
        .hemisphere_centers = .{ c - half_axis, c + half_axis },
        .radius = @abs(rnd(i, stream +% 7)) + 0.25,
    };
}

fn rnd_cylinder(i: usize, stream: u32) Cylinderf {
    const c = rnd_vec3(i, stream) * @as(Vec3, @splat(4));
    const half_axis = rnd_unit_vec3(i, stream +% 3);
    return .from_two_points_radius(c - half_axis, c + half_axis, @abs(rnd(i, stream +% 7)) + 0.25);
}

fn rnd_obb(i: usize, stream: u32) OBBf {
    return .from_orientation_and_half_extent(
        rnd_rigid_mat4(i, stream),
        @abs(rnd_vec3(i, stream +% 9)) + @as(Vec3, @splat(0.25)),
    );
}

fn rnd_affine(i: usize, stream: u32) Affine3 {
    const t = caliper.affine.translation(rnd_vec3(i, stream) * @as(Vec3, @splat(3)));
    const s = caliper.affine.scaling(@abs(rnd_vec3(i, stream +% 3)) + @as(Vec3, @splat(0.5)));
    return caliper.affine.times(t, s);
}

/// Every geometry query below early-outs. A batch where nothing hits would only ever time the
/// reject path and still print a plausible-looking number, so each generator alternates on this:
/// half the elements are placed to hit, half to miss.
inline fn is_hit(i: usize) bool {
    return i % 2 == 0;
}

/// The translation that moves `p` onto `target` when `is_hit(i)`, or far away otherwise.
fn hit_offset(i: usize, target: Vec3, p: Vec3) Vec3 {
    return if (is_hit(i)) target - p else @as(Vec3, @splat(100));
}

/// `target` itself when `is_hit(i)`, a point far from it otherwise.
fn point_at(i: usize, target: Vec3) Vec3 {
    return if (is_hit(i)) target else target + @as(Vec3, @splat(100));
}

// ── matrix ───────────────────────────────────────────────────────────────────

fn gen_mat4(i: usize) struct { Mat4 } {
    return .{ rnd_rigid_mat4(i, 0) };
}

fn gen_quat1(i: usize) struct { Vec4 } {
    return .{ rnd_quat(i, 0) };
}

fn gen_lookat(i: usize) struct { Vec3, Vec3, Vec3 } {
    const eye = rnd_vec3(i, 0) * @as(Vec3, @splat(5));
    // Keep eye and centre well apart: lookAt normalizes (centre - eye).
    return .{ eye, eye + rnd_unit_vec3(i, 4) * @as(Vec3, @splat(4)), .{ 0, 1, 0 } };
}

// ── quat ─────────────────────────────────────────────────────────────────────

fn gen_quat_pair(i: usize) struct { Vec4, Vec4 } {
    return .{ rnd_quat(i, 0), rnd_quat(i, 8) };
}

fn gen_quat_vec(i: usize) struct { Vec4, Vec3 } {
    return .{ rnd_quat(i, 0), rnd_vec3(i, 8) };
}

fn gen_quat_interp(i: usize) struct { Vec4, Vec4, f32 } {
    return .{ rnd_quat(i, 0), rnd_quat(i, 8), @abs(rnd(i, 16)) };
}

fn gen_axis_angle(i: usize) struct { Vec3, f32 } {
    return .{ rnd_unit_vec3(i, 0), rnd(i, 8) * std.math.pi };
}

// ── affine ───────────────────────────────────────────────────────────────────

fn gen_affine(i: usize) struct { Affine3 } {
    return .{ rnd_affine(i, 0) };
}

fn gen_affine_pair(i: usize) struct { Affine3, Affine3 } {
    return .{ rnd_affine(i, 0), rnd_affine(i, 16) };
}

fn gen_affine_vec(i: usize) struct { Affine3, Vec3 } {
    return .{ rnd_affine(i, 0), rnd_vec3(i, 16) };
}

// ── geometry: overlap ────────────────────────────────────────────────────────

// Broadphase shape: one query box against many candidates -- the pattern overlap_aabb_aabb_4
// exists for. The scalar and 4-wide rows share this box and these candidates, and both cover 1024
// boxes per run (the 4-wide packs 4 per call, so it registers 256 elements against the scalar's
// 1024), which is what makes the two timings directly comparable.
const query_box: AABBf = .{ .min = .{ -1, -1, -1 }, .max = .{ 1, 1, 1 } };

fn candidate_box(i: usize) AABBf {
    const b = rnd_aabb(i, 16);
    return b.translate(hit_offset(i, query_box.get_center(), b.get_center()));
}

fn gen_overlap_scalar(i: usize) struct { AABBf, AABBf } {
    return .{ query_box, candidate_box(i) };
}

fn gen_overlap_x4(i: usize) struct { AABBf, Vec4, Vec4, Vec4, Vec4, Vec4, Vec4 } {
    var lo: [3]Vec4 = .{ @splat(0), @splat(0), @splat(0) };
    var hi: [3]Vec4 = .{ @splat(0), @splat(0), @splat(0) };
    inline for (0..4) |k| {
        const b = candidate_box(i * 4 + k);
        inline for (0..3) |c| {
            lo[c][k] = b.min[c];
            hi[c][k] = b.max[c];
        }
    }
    return .{ query_box, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2] };
}

fn gen_sphere_pair(i: usize) struct { Spheref, Spheref } {
    const a = rnd_sphere(i, 0);
    const b = rnd_sphere(i, 16);
    return .{ a, b.translate(hit_offset(i, a.center, b.center)) };
}

fn gen_aabb_sphere(i: usize) struct { AABBf, Spheref } {
    const a = rnd_aabb(i, 0);
    const s = rnd_sphere(i, 16);
    return .{ a, s.translate(hit_offset(i, a.get_center(), s.center)) };
}

fn gen_aabb_plane(i: usize) struct { AABBf, Planef } {
    const a = rnd_aabb(i, 0);
    const n = rnd_unit_vec3(i, 16);
    // Hit: plane straight through the centre. Miss: shifted well clear along its own normal.
    const p = if (is_hit(i)) a.get_center() else a.get_center() + n * @as(Vec3, @splat(50));
    return .{ a, Planef.from_point_and_normal(p, n) };
}

/// A randomly-oriented OBB centred on `target` (or far from it). Built directly rather than via
/// `rnd_obb` + `modify_column`, which currently fails to compile (matrix.zig indexes a @Vector
/// with a runtime loop variable).
fn placed_obb(i: usize, stream: u32, target: Vec3) OBBf {
    const orientation = Mat4.identity
        .rotate(rnd(i, stream) * std.math.pi, rnd_unit_vec3(i, stream +% 1))
        .translate(point_at(i, target));
    return .from_orientation_and_half_extent(
        orientation,
        @abs(rnd_vec3(i, stream +% 9)) + @as(Vec3, @splat(0.25)),
    );
}

fn gen_obb_pair(i: usize) struct { OBBf, OBBf, f32 } {
    const a = rnd_obb(i, 0);
    return .{ a, placed_obb(i, 16, a.get_center()), 1.0e-6 };
}

fn gen_obb_aabb(i: usize) struct { OBBf, AABBf, f32 } {
    const box = rnd_aabb(i, 0);
    return .{ placed_obb(i, 16, box.get_center()), box, 1.0e-6 };
}

fn gen_aabb_triangle(i: usize) struct { AABBf, Vec3, Vec3, Vec3 } {
    const box = rnd_aabb(i, 0);
    const base = point_at(i, box.get_center());
    return .{ box, base + rnd_vec3(i, 16), base + rnd_vec3(i, 20), base + rnd_vec3(i, 24) };
}

// ── geometry: contains ───────────────────────────────────────────────────────

fn gen_aabb_point(i: usize) struct { AABBf, Vec3 } {
    const a = rnd_aabb(i, 0);
    return .{ a, point_at(i, a.get_center()) };
}

fn gen_sphere_point(i: usize) struct { Spheref, Vec3 } {
    const a = rnd_sphere(i, 0);
    return .{ a, point_at(i, a.center) };
}

fn gen_capsule_point(i: usize) struct { Capsulef, Vec3 } {
    const a = rnd_capsule(i, 0);
    return .{ a, point_at(i, a.center()) };
}

fn gen_obb_point(i: usize) struct { OBBf, Vec3 } {
    const a = rnd_obb(i, 0);
    return .{ a, point_at(i, a.get_center()) };
}

// ── geometry: ray ────────────────────────────────────────────────────────────

/// A ray `dist` away from `target`, pointing at it when `is_hit(i)` and directly away otherwise.
fn ray_at(i: usize, stream: u32, target: Vec3, dist: f32) struct { Vec3, Vec3 } {
    const u = rnd_unit_vec3(i, stream);
    return .{ target + u * @as(Vec3, @splat(dist)), if (is_hit(i)) -u else u };
}

fn gen_ray_aabb(i: usize) struct { AABBf, Vec3, InvDirf } {
    const box = rnd_aabb(i, 0);
    const r = ray_at(i, 16, box.get_center(), 10);
    // InvDirection is precomputed, as a caller sweeping many boxes would hoist it, so the timing
    // isolates the slab test rather than the reciprocal.
    return .{ box, r[0], InvDirf.from_direction(r[1]) };
}

fn gen_ray_sphere(i: usize) struct { Spheref, Vec3, Vec3 } {
    const s = rnd_sphere(i, 0);
    const r = ray_at(i, 16, s.center, 10);
    return .{ s, r[0], r[1] };
}

fn gen_ray_capsule(i: usize) struct { Capsulef, Vec3, Vec3 } {
    const c = rnd_capsule(i, 0);
    const r = ray_at(i, 16, c.center(), 10);
    return .{ c, r[0], r[1] };
}

fn gen_ray_cylinder(i: usize) struct { Cylinderf, Vec3, Vec3 } {
    const c = rnd_cylinder(i, 0);
    const r = ray_at(i, 16, c.center(), 10);
    return .{ c, r[0], r[1] };
}

fn gen_ray_triangle(i: usize) struct { Vec3, Vec3, Vec3, Vec3, Vec3 } {
    const v0 = rnd_vec3(i, 0) * @as(Vec3, @splat(4));
    const v1 = v0 + rnd_vec3(i, 4);
    const v2 = v0 + rnd_vec3(i, 8);
    const r = ray_at(i, 16, (v0 + v1 + v2) / @as(Vec3, @splat(3)), 10);
    return .{ r[0], r[1], v0, v1, v2 };
}

/// `ray_triangle` takes its scalar type as a comptime argument, which cannot live in a runtime
/// input tuple -- pin it to f32.
fn ray_triangle_f32(origin: Vec3, direction: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) f32 {
    return caliper.geom.ray.ray_triangle(f32, origin, direction, v0, v1, v2);
}

// ── geometry: closest_point / GJK simplex ────────────────────────────────────

// The query point is drawn from a wider range than the simplex, so it usually lands outside and
// the vertex/edge/face regions all get exercised rather than just the interior case.

fn gen_cp_line(i: usize) struct { Vec3, Vec3, Vec3 } {
    return .{ rnd_vec3(i, 0) * @as(Vec3, @splat(3)), rnd_vec3(i, 4), rnd_vec3(i, 8) };
}

fn gen_cp_triangle(i: usize) struct { Vec3, Vec3, Vec3, Vec3 } {
    return .{ rnd_vec3(i, 0) * @as(Vec3, @splat(3)), rnd_vec3(i, 4), rnd_vec3(i, 8), rnd_vec3(i, 12) };
}

fn gen_cp_tetra(i: usize) struct { Vec3, Vec3, Vec3, Vec3, Vec3 } {
    return .{ rnd_vec3(i, 0) * @as(Vec3, @splat(3)), rnd_vec3(i, 4), rnd_vec3(i, 8), rnd_vec3(i, 12), rnd_vec3(i, 16) };
}

// Same comptime-type-argument shim as ray_triangle_f32.
fn cp_line_f32(p: Vec3, a: Vec3, b: Vec3) Vec3 {
    return caliper.geom.closest_point.closest_point_on_line_to(f32, p, a, b);
}

fn cp_triangle_f32(p: Vec3, a: Vec3, b: Vec3, c: Vec3) Vec3 {
    return caliper.geom.closest_point.closest_point_on_triangle_to(f32, p, a, b, c);
}

fn cp_tetra_f32(p: Vec3, a: Vec3, b: Vec3, c: Vec3, d: Vec3) Vec3 {
    return caliper.geom.closest_point.closest_point_on_tetrahedron_to(f32, p, a, b, c, d);
}

// ── geometry: frustum culling ────────────────────────────────────────────────

/// Built once in `main()`. Kept in a global rather than in every batch element so the 6-plane
/// frustum isn't duplicated 1024 times, which would push the batch clean out of cache.
var bench_frustum: caliper.geom.Frustumf32 = undefined;

fn frustum_aabb(box: AABBf) bool {
    return bench_frustum.intersect_aabb(box);
}

fn frustum_sphere(s: Spheref) bool {
    return bench_frustum.intersect_sphere(s);
}

fn frustum_capsule(c: Capsulef) bool {
    return bench_frustum.intersect_capsule(c);
}

// The camera looks down -z, so `visible` sits in front of it and `culled` far off to the side.
const visible_at: Vec3 = .{ 0, 0, -20 };
const culled_at: Vec3 = .{ 500, 0, -20 };

fn gen_cull_aabb(i: usize) struct { AABBf } {
    const b = rnd_aabb(i, 0);
    const target = if (is_hit(i)) visible_at else culled_at;
    return .{ b.translate(target - b.get_center()) };
}

fn gen_cull_sphere(i: usize) struct { Spheref } {
    const s = rnd_sphere(i, 0);
    const target = if (is_hit(i)) visible_at else culled_at;
    return .{ s.translate(target - s.center) };
}

fn gen_cull_capsule(i: usize) struct { Capsulef } {
    const c = rnd_capsule(i, 0);
    const target = if (is_hit(i)) visible_at else culled_at;
    const d = target - c.center();
    return .{ .{ .hemisphere_centers = .{ c.hemisphere_centers[0] + d, c.hemisphere_centers[1] + d }, .radius = c.radius } };
}

// ── geometry: support functions ──────────────────────────────────────────────

fn gen_aabb_dir(i: usize) struct { AABBf, Vec3 } {
    return .{ rnd_aabb(i, 0), rnd_unit_vec3(i, 16) };
}

fn gen_sphere_dir(i: usize) struct { Spheref, Vec3 } {
    return .{ rnd_sphere(i, 0), rnd_unit_vec3(i, 16) };
}

fn gen_capsule_dir(i: usize) struct { Capsulef, Vec3 } {
    return .{ rnd_capsule(i, 0), rnd_unit_vec3(i, 16) };
}

fn gen_cylinder_dir(i: usize) struct { Cylinderf, Vec3 } {
    return .{ rnd_cylinder(i, 0), rnd_unit_vec3(i, 16) };
}

fn gen_obb_dir(i: usize) struct { OBBf, Vec3 } {
    return .{ rnd_obb(i, 0), rnd_unit_vec3(i, 16) };
}

// ── geometry: GJK ────────────────────────────────────────────────────────────

// GJK doesn't fit `batch`: the solver is mutable and the queries take it plus an in/out separating
// axis by pointer. The solver is a `run` local (never a field) because `run` may not mutate `self`
// -- see the note on the batching harness above. Sphere/OBB are used rather than AABB: GJK needs a
// *maximizing* get_support, and AABB's minimizes (see frustum.zig).

const GJKf = caliper.geom.GJK(f32);
const gjk_tolerance: f32 = 1.0e-4;

fn gjk_pair(i: usize) struct { Spheref, OBBf } {
    const a = rnd_sphere(i, 0);
    return .{ a, placed_obb(i, 16, a.center) };
}

fn bench_gjk_intersects(comptime size: usize) type {
    return struct {
        pairs: [size]struct { Spheref, OBBf },
        fn init() @This() {
            var in: [size]struct { Spheref, OBBf } = undefined;
            for (0..size) |i| in[i] = gjk_pair(i);
            return .{ .pairs = in };
        }
        pub fn run(self: *@This(), _: std.mem.Allocator) void {
            var solver: GJKf = .{};
            for (self.pairs) |p| {
                var v: Vec3 = .{ 1, 0, 0 }; // initial separating-axis guess; must be non-zero
                std.mem.doNotOptimizeAway(@call(.never_inline, GJKf.intersects, .{ &solver, p[0], p[1], gjk_tolerance, &v }));
            }
        }
    };
}

fn bench_gjk_closest_points(comptime size: usize) type {
    return struct {
        pairs: [size]struct { Spheref, OBBf },
        fn init() @This() {
            var in: [size]struct { Spheref, OBBf } = undefined;
            for (0..size) |i| in[i] = gjk_pair(i);
            return .{ .pairs = in };
        }
        pub fn run(self: *@This(), _: std.mem.Allocator) void {
            var solver: GJKf = .{};
            for (self.pairs) |p| {
                var v: Vec3 = .{ 1, 0, 0 };
                var pa: Vec3 = undefined;
                var pb: Vec3 = undefined;
                std.mem.doNotOptimizeAway(@call(.never_inline, GJKf.get_closest_points, .{
                    &solver, p[0], p[1], gjk_tolerance, std.math.floatMax(f32), &v, &pa, &pb,
                }));
            }
        }
    };
}

fn bench_gjk_cast_ray(comptime size: usize) type {
    return struct {
        shapes: [size]struct { OBBf, Vec3, Vec3 },
        fn init() @This() {
            var in: [size]struct { OBBf, Vec3, Vec3 } = undefined;
            for (0..size) |i| {
                const o = rnd_obb(i, 0);
                const u = rnd_unit_vec3(i, 16);
                // lambda is a fraction of `direction`, so the ray must span the shape within [0,1].
                const origin = o.get_center() + u * @as(Vec3, @splat(5));
                const dir = if (is_hit(i)) -u * @as(Vec3, @splat(10)) else u * @as(Vec3, @splat(10));
                in[i] = .{ o, origin, dir };
            }
            return .{ .shapes = in };
        }
        pub fn run(self: *@This(), _: std.mem.Allocator) void {
            var solver: GJKf = .{};
            for (self.shapes) |s| {
                var lambda: f32 = 1.0;
                std.mem.doNotOptimizeAway(@call(.never_inline, GJKf.cast_ray, .{ &solver, s[1], s[2], gjk_tolerance, s[0], &lambda }));
            }
        }
    };
}

pub fn main() !void {
    const io = std.Io.Threaded.global_single_threaded.io();
    const stdout: std.Io.File = .stdout();

    // A 60-degree fov camera at the origin looking down -z. Spelled out rather than via
    // caliper.to_radians, which currently fails to compile (root.zig divides comptime_float by
    // comptime_int).
    bench_frustum = .from_view_projection(
        Mat4.perspective(60.0 * std.math.pi / 180.0, 16.0 / 9.0, 0.1, 1000)
            .mul(Mat4.lookAt(.{ 0, 0, 0 }, .{ 0, 0, -1 }, .{ 0, 1, 0 })),
    );

    var bench = zbench.Benchmark.init(std.heap.page_allocator, .{});
    defer bench.deinit();

    try bench.addParam("Multiple 4x4 matrix multiplication", &benchmark_multiply(4).init(), .{
        .iterations = 256,
    });
    try bench.addParam("Multiple 512x512 matrix multiplication", &benchmark_multiply(512).init(), .{
        .iterations = 256,
    });
    try bench.addParam("Sin/Cos", &benchmark_sin_cos_system(256).init(), .{
        .iterations = 256,
    });
    try bench.addParam("Sin/Cos vectorized", &bench_sin_cos_fused(256).init(), .{
        .iterations = 256,
    });
    try bench.addParam("Sin/Cos array (chunked)", &bench_sin_cos_array(256).init(), .{
        .iterations = 256,
    });

    // Chunked array path vs single wide-@Vector op, at the motivating length 245.
    try bench.addParam("norm [245] array (chunked)", &bench_norm_array(245).init(), .{
        .iterations = 4096,
    });
    try bench.addParam("norm @Vector(245)", &bench_norm_vector(245).init(), .{
        .iterations = 4096,
    });
    try bench.addParam("dot [245] array (chunked)", &bench_dot_array(245).init(), .{
        .iterations = 4096,
    });
    try bench.addParam("dot @Vector(245)", &bench_dot_vector(245).init(), .{
        .iterations = 4096,
    });
    try bench.addParam("normalize [245] array (chunked)", &bench_normalize_array(245).init(), .{
        .iterations = 4096,
    });
    try bench.addParam("normalize @Vector(245)", &bench_normalize_vector(245).init(), .{
        .iterations = 4096,
    });

    // Everything below runs its op over a batch of inputs (see `batch`), so one "iteration" is
    // N ops. N is 1024 for elements up to ~32 bytes and 256 for the bigger ones, to keep the
    // batch inside L1.
    const it: u32 = 256;

    // matrix
    try bench.addParam("Mat4 inverse x256", &batch(256, gen_mat4, Mat4.inverse).init(), .{ .iterations = it });
    try bench.addParam("Mat4 inverse_ortho x256", &batch(256, gen_mat4, Mat4.inverse_ortho).init(), .{ .iterations = it });
    try bench.addParam("Mat4 transpose x256", &batch(256, gen_mat4, Mat4.transpose).init(), .{ .iterations = it });
    try bench.addParam("Mat4 determinant x256", &batch(256, gen_mat4, Mat4.determinant).init(), .{ .iterations = it });
    try bench.addParam("Mat4 decompose x256", &batch(256, gen_mat4, Mat4.decompose).init(), .{ .iterations = it });
    try bench.addParam("Mat4 from_quat x1024", &batch(1024, gen_quat1, Mat4.from_quat).init(), .{ .iterations = it });
    try bench.addParam("Mat4 lookAt x256", &batch(256, gen_lookat, Mat4.lookAt).init(), .{ .iterations = it });

    // quat
    try bench.addParam("quat mul x1024", &batch(1024, gen_quat_pair, caliper.quat.mul).init(), .{ .iterations = it });
    try bench.addParam("quat rotate_vector x1024", &batch(1024, gen_quat_vec, caliper.quat.rotate_vector).init(), .{ .iterations = it });
    try bench.addParam("quat rotate_vector_inv x1024", &batch(1024, gen_quat_vec, caliper.quat.rotate_vector_inv).init(), .{ .iterations = it });
    try bench.addParam("quat slerp x256", &batch(256, gen_quat_interp, caliper.quat.slerp).init(), .{ .iterations = it });
    try bench.addParam("quat lerp x256", &batch(256, gen_quat_interp, caliper.quat.lerp).init(), .{ .iterations = it });
    try bench.addParam("quat inverse x1024", &batch(1024, gen_quat1, caliper.quat.inverse).init(), .{ .iterations = it });
    try bench.addParam("quat to_matrix x1024", &batch(1024, gen_quat1, caliper.quat.to_matrix).init(), .{ .iterations = it });
    try bench.addParam("quat from_matrix x256", &batch(256, gen_mat4, caliper.quat.from_matrix).init(), .{ .iterations = it });
    try bench.addParam("quat from_rotation x1024", &batch(1024, gen_axis_angle, caliper.quat.from_rotation).init(), .{ .iterations = it });

    // affine (3x4)
    try bench.addParam("affine times x256", &batch(256, gen_affine_pair, caliper.affine.times).init(), .{ .iterations = it });
    try bench.addParam("affine times_point x256", &batch(256, gen_affine_vec, caliper.affine.times_point).init(), .{ .iterations = it });
    try bench.addParam("affine times_dir x256", &batch(256, gen_affine_vec, caliper.affine.times_dir).init(), .{ .iterations = it });
    try bench.addParam("affine get_scale x256", &batch(256, gen_affine, caliper.affine.get_scale).init(), .{ .iterations = it });

    // geometry: overlap. The two aabb_aabb rows cover the same 1024 candidate boxes against the
    // same query box, so their totals compare directly (4-wide should win).
    try bench.addParam("overlap_aabb_aabb x1024 (scalar)", &batch(1024, gen_overlap_scalar, caliper.geom.overlap.overlap_aabb_aabb).init(), .{ .iterations = it });
    try bench.addParam("overlap_aabb_aabb_4 x1024 (SIMD 4-wide)", &batch(256, gen_overlap_x4, caliper.geom.overlap.overlap_aabb_aabb_4).init(), .{ .iterations = it });
    try bench.addParam("overlap_sphere_sphere x1024", &batch(1024, gen_sphere_pair, caliper.geom.overlap.overlap_sphere_sphere).init(), .{ .iterations = it });
    try bench.addParam("overlap_aabb_sphere x256", &batch(256, gen_aabb_sphere, caliper.geom.overlap.overlap_aabb_sphere).init(), .{ .iterations = it });
    try bench.addParam("overlap_aabb_plane x256", &batch(256, gen_aabb_plane, caliper.geom.overlap.overlap_aabb_plane).init(), .{ .iterations = it });
    try bench.addParam("overlap_aabb_triangle x256 (SAT)", &batch(256, gen_aabb_triangle, caliper.geom.overlap.overlap_aabb_triangle).init(), .{ .iterations = it });
    try bench.addParam("overlap_obb_obb x256 (SAT)", &batch(256, gen_obb_pair, caliper.geom.overlap.overlap_obb_obb).init(), .{ .iterations = it });
    try bench.addParam("overlap_obb_aabb x256 (SAT)", &batch(256, gen_obb_aabb, caliper.geom.overlap.overlap_obb_aabb).init(), .{ .iterations = it });

    // geometry: contains
    try bench.addParam("aabb_contains_point x1024", &batch(1024, gen_aabb_point, caliper.geom.contains.aabb_contains_point).init(), .{ .iterations = it });
    try bench.addParam("sphere_contains_point x1024", &batch(1024, gen_sphere_point, caliper.geom.contains.sphere_contains_point).init(), .{ .iterations = it });
    try bench.addParam("capsule_contains_point x1024", &batch(1024, gen_capsule_point, caliper.geom.contains.capsule_contains_point).init(), .{ .iterations = it });
    try bench.addParam("obb_contains_point x256", &batch(256, gen_obb_point, caliper.geom.contains.obb_contains_point).init(), .{ .iterations = it });

    // geometry: ray
    try bench.addParam("ray_aabb x256", &batch(256, gen_ray_aabb, caliper.geom.ray.ray_aabb).init(), .{ .iterations = it });
    try bench.addParam("ray_aabb_with_enter_exit x256", &batch(256, gen_ray_aabb, caliper.geom.ray.ray_aabb_with_enter_exit).init(), .{ .iterations = it });
    try bench.addParam("ray_sphere x256", &batch(256, gen_ray_sphere, caliper.geom.ray.ray_sphere).init(), .{ .iterations = it });
    try bench.addParam("ray_triangle x256 (Moller-Trumbore)", &batch(256, gen_ray_triangle, ray_triangle_f32).init(), .{ .iterations = it });
    try bench.addParam("ray_capsule x256", &batch(256, gen_ray_capsule, caliper.geom.ray.ray_capsule).init(), .{ .iterations = it });
    try bench.addParam("ray_cylinder x256", &batch(256, gen_ray_cylinder, caliper.geom.ray.ray_cylinder).init(), .{ .iterations = it });

    // geometry: closest point (GJK simplex internals)
    try bench.addParam("closest_point_on_line x256", &batch(256, gen_cp_line, cp_line_f32).init(), .{ .iterations = it });
    try bench.addParam("closest_point_on_triangle x256", &batch(256, gen_cp_triangle, cp_triangle_f32).init(), .{ .iterations = it });
    try bench.addParam("closest_point_on_tetrahedron x256", &batch(256, gen_cp_tetra, cp_tetra_f32).init(), .{ .iterations = it });

    // geometry: frustum culling
    try bench.addParam("frustum intersect_aabb x1024", &batch(1024, gen_cull_aabb, frustum_aabb).init(), .{ .iterations = it });
    try bench.addParam("frustum intersect_sphere x1024", &batch(1024, gen_cull_sphere, frustum_sphere).init(), .{ .iterations = it });
    try bench.addParam("frustum intersect_capsule x1024", &batch(1024, gen_cull_capsule, frustum_capsule).init(), .{ .iterations = it });

    // geometry: support functions
    try bench.addParam("AABB get_support x1024", &batch(1024, gen_aabb_dir, AABBf.get_support).init(), .{ .iterations = it });
    try bench.addParam("Sphere get_support x1024", &batch(1024, gen_sphere_dir, Spheref.get_support).init(), .{ .iterations = it });
    try bench.addParam("Capsule get_support x1024", &batch(1024, gen_capsule_dir, Capsulef.get_support).init(), .{ .iterations = it });
    try bench.addParam("Cylinder get_support x1024", &batch(1024, gen_cylinder_dir, Cylinderf.get_support).init(), .{ .iterations = it });
    try bench.addParam("OBB get_support x256", &batch(256, gen_obb_dir, OBBf.get_support).init(), .{ .iterations = it });

    // geometry: GJK (sphere vs OBB)
    try bench.addParam("GJK intersects x256", &bench_gjk_intersects(256).init(), .{ .iterations = it });
    try bench.addParam("GJK get_closest_points x256", &bench_gjk_closest_points(256).init(), .{ .iterations = it });
    try bench.addParam("GJK cast_ray x256", &bench_gjk_cast_ray(256).init(), .{ .iterations = it });

    try bench.run(io, stdout);
}

