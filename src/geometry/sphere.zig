const zml = @import("../root.zig");
const geometry = zml.geom;

pub fn Sphere(comptime T: type) type {
    return struct {
        pub const child: type = T;
        pub const primative_type: geometry.Primative = .Sphere;

        pub const Self = @This();
        pub const empty: Self = .{
            .center = @Vector(3, T){0, 0, 0} ,
            .radius = 0,
        };

        center: @Vector(3, T),
        radius: T,

        pub fn from_center_radius(center: @Vector(3, T), radius: T) Sphere(T) {
            return .{
                .center = center,
                .radius = radius,
            };
        }
    };
}
