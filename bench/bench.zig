const std = @import("std");
const zbench = @import("zbench");
const zml = @import("zml");

fn benchmark_multiply(comptime size: usize) type {
    return struct {
        const Matrix512 = zml.Mat(f32, size, size);
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
           std.mem.doNotOptimizeAway(@call(.never_inline, zml.vec.sin_cos, .{self.angles}));
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
            std.mem.doNotOptimizeAway(@call(.never_inline, zml.vec.sin_cos, .{self.angles}));
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
            std.mem.doNotOptimizeAway(@call(.never_inline, zml.vec.norm_adv, .{ self.data, 32 }));
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
            std.mem.doNotOptimizeAway(@call(.never_inline, zml.vec.norm_adv, .{ self.data, 32 }));
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
            std.mem.doNotOptimizeAway(@call(.never_inline, zml.vec.dot, .{ self.a, self.b }));
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
            std.mem.doNotOptimizeAway(@call(.never_inline, zml.vec.dot, .{ self.a, self.b }));
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
            std.mem.doNotOptimizeAway(@call(.never_inline, zml.vec.normalize, .{self.data}));
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
            std.mem.doNotOptimizeAway(@call(.never_inline, zml.vec.normalize, .{self.data}));
        }
    };
}

pub fn main() !void {
    const io = std.Io.Threaded.global_single_threaded.io();
    const stdout: std.Io.File = .stdout();

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

    try bench.run(io, stdout);
}

