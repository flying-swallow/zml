const std = @import("std");
const meta = @import("../meta.zig");

pub fn BufferView(comptime T: type) type {
    return struct {
        pub const inner_type = [meta.lengthOf(T)]meta.Child(T);
        pub const num_elements = meta.lengthOf(T);
    };
}
