const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const root_module = b.addModule("zml", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    {
        const tests = b.addTest(.{ .name = "test", .root_module = root_module });
        const run_tests = b.addRunArtifact(tests);
        const test_step = b.step("test", "Run unit tests");
        test_step.dependOn(&run_tests.step);
    }
    
    if (b.isRoot()) {
        if (b.lazyDependency("zbench", .{
            .target = target,
            .optimize = optimize,
        })) |zbench_dep| {
            const bench = b.addExecutable(.{
                .name = "bench",
                .root_module = b.createModule(.{
                    .root_source_file = b.path("bench/bench.zig"),
                    .target = target,
                    .optimize = optimize,
                    .imports = &.{
                        .{ .name = "zml", .module = root_module },
                        .{ .name = "zbench", .module = zbench_dep.module("zbench") },
                    },
                }),
            });
            const bench_step = b.step("bench", "run benchmark");
            const bench_cmd = b.addRunArtifact(bench);
            bench_step.dependOn(&bench_cmd.step);
        }
    }

}
