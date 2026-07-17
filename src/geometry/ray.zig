const std = @import("std");
const zml = @import("../root.zig");

pub fn InvDirection(comptime T: type) type {
    return struct {
        inv_direction: @Vector(3, T), // 1 / ray direction
        is_parallel: @Vector(3, bool), // true if ray direction is close to zero

        pub fn from_direction(direction: @Vector(3, T)) @This() {
            return .{
                .is_parallel = @abs(direction) < @as(@Vector(3, T), @splat(1.0e-20)),
                .inv_direction = @as(@Vector(3, T), @splat(1)) / direction,
            };
        }
    };
}

/// Ray vs finite capped cylinder. Returns the fraction along `direction` of the first
/// hit (0 if the ray starts inside), or std.math.floatMax(T) if there is no forward hit.
pub fn ray_cylinder(cylinder: anytype, origin: @Vector(3, @TypeOf(cylinder).child), direction: @Vector(3, @TypeOf(cylinder).child)) @TypeOf(cylinder).child {
    comptime {
        std.debug.assert(@TypeOf(cylinder).primative_type == .Cylinder); // ensure cylinder type
    }
    const T = @TypeOf(cylinder).child;
    const V = @Vector(3, T);

    const p0 = cylinder.end_points[0];
    const u = cylinder.axis(); // unit axis p0 -> p1
    const h = cylinder.get_height();
    const r_sq = cylinder.radius * cylinder.radius;

    // Work relative to p0, splitting each vector into axial (along u) and radial parts.
    const o = origin - p0;
    const o_axial = zml.vec.dot(o, u);
    const d_axial = zml.vec.dot(direction, u);
    const o_perp = o - u * @as(V, @splat(o_axial));
    const d_perp = direction - u * @as(V, @splat(d_axial));

    // Origin already inside the finite cylinder.
    if (zml.vec.dot(o_perp, o_perp) <= r_sq and o_axial >= 0 and o_axial <= h) return 0;

    var best = std.math.floatMax(T);

    // Lateral (curved) surface: |o_perp + t*d_perp|^2 = r^2, keeping hits within [0, h].
    const a = zml.vec.dot(d_perp, d_perp);
    const b = 2 * zml.vec.dot(o_perp, d_perp);
    const c = zml.vec.dot(o_perp, o_perp) - r_sq;
    const roots = zml.find_roots(T, a, b, c);
    for (0..roots.num_roots) |i| {
        const t = roots.roots[i];
        if (t >= 0 and t < best) {
            const axial_at = o_axial + t * d_axial;
            if (axial_at >= 0 and axial_at <= h) best = t;
        }
    }

    // End caps: planes at axial = 0 and axial = h, accepted where the hit is within the radius.
    if (d_axial != 0) {
        const caps = [2]T{ 0, h };
        for (caps) |cap| {
            const t = (cap - o_axial) / d_axial;
            if (t >= 0 and t < best) {
                const radial = o_perp + d_perp * @as(V, @splat(t));
                if (zml.vec.dot(radial, radial) <= r_sq) best = t;
            }
        }
    }

    return best;
}

/// Ray vs sphere. Returns the fraction along `direction` of the first hit (0 if the ray starts
/// inside the sphere), or std.math.floatMax(T) if there is no forward hit. `direction` need not
/// be normalized. Adapted from JoltPhysics RaySphere.
pub fn ray_sphere(sphere: anytype, origin: @Vector(3, @TypeOf(sphere).child), direction: @Vector(3, @TypeOf(sphere).child)) @TypeOf(sphere).child {
    comptime {
        std.debug.assert(@TypeOf(sphere).primative_type == .Sphere); // ensure sphere type
    }
    const T = @TypeOf(sphere).child;

    // Solve |origin + t*direction - center|^2 = radius^2 for t.
    const center_origin = origin - sphere.center;
    const a = zml.vec.dot(direction, direction);
    const b = 2 * zml.vec.dot(direction, center_origin);
    const c = zml.vec.dot(center_origin, center_origin) - sphere.radius * sphere.radius;
    const roots = zml.find_roots(T, a, b, c);

    // No real roots: the (possibly degenerate) ray never crosses the surface. Inside -> 0, else miss.
    if (roots.num_roots == 0) return if (c <= 0) 0 else std.math.floatMax(T);

    // Sort so the smallest fraction (ray entering the sphere) is first.
    var f1 = roots.roots[0];
    var f2 = roots.roots[1];
    if (f1 > f2) std.mem.swap(T, &f1, &f2);

    if (f1 >= 0) return f1; // Entering the sphere ahead of the origin.
    if (f2 >= 0) return 0; // Origin is inside the sphere.
    return std.math.floatMax(T); // Sphere is entirely behind the origin.
}

/// Ray vs capsule. Returns the fraction along `direction` of the first hit (0 if the ray starts
/// inside the capsule), or std.math.floatMax(T) if there is no forward hit. `direction` need not
/// be normalized. Adapted from JoltPhysics RayCapsule, generalised to the arbitrary-axis capsule.
pub fn ray_capsule(capsule: anytype, origin: @Vector(3, @TypeOf(capsule).child), direction: @Vector(3, @TypeOf(capsule).child)) @TypeOf(capsule).child {
    comptime {
        std.debug.assert(@TypeOf(capsule).primative_type == .Capsule); // ensure capsule type
    }
    const T = @TypeOf(capsule).child;

    // The capsule is the union of the finite cylinder between the two hemisphere centres and a
    // sphere at each centre. The first entry into that union is the min over the three parts
    // (each returns floatMax on a miss and 0 when the origin is already inside it).
    const cyl = zml.geom.Cylinder(T).from_two_points_radius(capsule.hemisphere_centers[0], capsule.hemisphere_centers[1], capsule.radius);
    const s0 = zml.geom.Sphere(T).from_center_radius(capsule.hemisphere_centers[0], capsule.radius);
    const s1 = zml.geom.Sphere(T).from_center_radius(capsule.hemisphere_centers[1], capsule.radius);
    return @min(ray_cylinder(cyl, origin, direction), @min(ray_sphere(s0, origin, direction), ray_sphere(s1, origin, direction)));
}

// Ray - Axis Aligned Bounding Box intersection
// Note: Can return negative t values if the ray origin is inside the AABB
// return std.math.floatMax(T) if no hit
pub fn ray_aabb(aabb: anytype, origin: @Vector(3, @TypeOf(aabb).child), invDir: InvDirection(@TypeOf(aabb).child)) std.meta.Float(@bitSizeOf(@TypeOf(aabb).child)) {
    comptime {
        std.debug.assert(@TypeOf(aabb).primative_type == .AABB); // ensure aabb type
    }
    const flt_min = @as(@Vector(3, @TypeOf(aabb).child), @splat(-std.math.floatMax(@TypeOf(aabb).child)));
    const flt_max = @as(@Vector(3, @TypeOf(aabb).child), @splat(std.math.floatMax(@TypeOf(aabb).child)));

    // Test against all three axes simultaneously.
    const t1 = (aabb.min - origin) * invDir.inv_direction;
    const t2 = (aabb.max - origin) * invDir.inv_direction;

    // Compute the max of min(t1,t2) and the min of max(t1,t2) ensuring we don't
    // use the results from any directions parallel to the slab.
    var t_min = @select(@TypeOf(aabb).child, invDir.is_parallel, flt_min, @min(t1, t2));
    var t_max = @select(@TypeOf(aabb).child, invDir.is_parallel, flt_max, @max(t1, t2));

    // t_min.xyz = maximum(t_min.x, t_min.y, t_min.z);
    t_min = @max(t_min, @shuffle(@TypeOf(aabb).child, t_min, t_min, [3]u8{ 1, 2, 0 }));
    t_min = @max(t_min, @shuffle(@TypeOf(aabb).child, t_min, t_min, [3]u8{ 2, 0, 1 }));

    // t_max.xyz = minimum(t_max.x, t_max.y, t_max.z);
    t_max = @min(t_max, @shuffle(@TypeOf(aabb).child, t_max, t_max, [3]u8{ 1, 2, 0 }));
    t_max = @min(t_max, @shuffle(@TypeOf(aabb).child, t_max, t_max, [3]u8{ 2, 0, 1 }));

    // if (t_min > t_max) return FLT_MAX;
    var no_intersection = t_min > t_max;

    // if (t_max < 0.0f) return FLT_MAX;
    no_intersection = no_intersection | (t_max < @as(@TypeOf(t_max), @splat(0)));

    // if (inInvDirection.mIsParallel && !(Min <= inOrigin && inOrigin <= Max)) return FLT_MAX; else return t_min;
    const no_parallel_overlap = (origin < aabb.min) | (origin > aabb.max);
    no_intersection = no_intersection | (invDir.is_parallel & no_parallel_overlap);
    no_intersection = no_intersection | @as(@TypeOf(no_intersection), @splat(no_intersection[1]));
    no_intersection = no_intersection | @as(@TypeOf(no_intersection), @splat(no_intersection[2]));

    return @select(@TypeOf(aabb).child, no_intersection, flt_max, t_min)[0];
}

// Ray - Axis Aligned Bounding Box intersection
// Note: Can return negative t values if the ray origin is inside the AABB
// return -std.math.floatMax(T) and std.math.floatMax(T) if no hit 
pub fn ray_aabb_with_enter_exit(aabb: anytype, origin: @Vector(3, @TypeOf(aabb).child), invDir: InvDirection(@TypeOf(aabb).child)) struct {
    min: std.meta.Float(@bitSizeOf(@TypeOf(aabb).child)),
    max: std.meta.Float(@bitSizeOf(@TypeOf(aabb).child))
} {
    comptime {
        std.debug.assert(@TypeOf(aabb).primative_type == .AABB); // ensure aabb type
    }
    const flt_min = @as(@Vector(3, @TypeOf(aabb).child), @splat(-std.math.floatMax(@TypeOf(aabb).child)));
    const flt_max = @as(@Vector(3, @TypeOf(aabb).child), @splat(std.math.floatMax(@TypeOf(aabb).child)));

    // Test against all three axes simultaneously.
    const t1 = (aabb.min - origin) * invDir.inv_direction;
    const t2 = (aabb.max - origin) * invDir.inv_direction;

    // Compute the max of min(t1,t2) and the min of max(t1,t2) ensuring we don't
    // use the results from any directions parallel to the slab.
    var t_min = @select(@TypeOf(aabb).child, invDir.is_parallel, flt_min, @min(t1, t2));
    var t_max = @select(@TypeOf(aabb).child, invDir.is_parallel, flt_max, @max(t1, t2));

    // t_min.xyz = maximum(t_min.x, t_min.y, t_min.z);
    t_min = @max(t_min, @shuffle(@TypeOf(aabb).child, t_min, t_min, [3]u8{ 1, 2, 0 }));
    t_min = @max(t_min, @shuffle(@TypeOf(aabb).child, t_min, t_min, [3]u8{ 2, 0, 1 }));

    // t_max.xyz = minimum(t_max.x, t_max.y, t_max.z);
    t_max = @min(t_max, @shuffle(@TypeOf(aabb).child, t_max, t_max, [3]u8{ 1, 2, 0 }));
    t_max = @min(t_max, @shuffle(@TypeOf(aabb).child, t_max, t_max, [3]u8{ 2, 0, 1 }));

    // if (t_min > t_max) return FLT_MAX;
    var no_intersection = t_min > t_max;

    // if (t_max < 0.0f) return FLT_MAX;
    no_intersection = no_intersection | (t_max < @as(@TypeOf(t_max), @splat(0)));

    // if (inInvDirection.mIsParallel && !(Min <= inOrigin && inOrigin <= Max)) return FLT_MAX; else return t_min;
    const no_parallel_overlap = (origin < aabb.min) | (origin > aabb.max);
    no_intersection = no_intersection | (invDir.is_parallel & no_parallel_overlap);
    no_intersection = no_intersection | @as(@TypeOf(no_intersection), @splat(no_intersection[1]));
    no_intersection = no_intersection | @as(@TypeOf(no_intersection), @splat(no_intersection[2]));

    return .{
        .min = @select(@TypeOf(aabb).child, no_intersection, flt_max, t_min)[0],
        .max = @select(@TypeOf(aabb).child, no_intersection, flt_max, t_max)[0],
    };

}

/// Intersect ray with triangle, returns closest point or FLT_MAX if no hit (branch less version)
/// Adapted from: http://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
pub fn ray_triangle(comptime T: type, origin: @Vector(3, T), direction: @Vector(3, T), v0: @Vector(3, T), v1: @Vector(3, T), v2: @Vector(3, T)) T {
    const epsilon: @Vector(3, T) = @as(@Vector(3, T), @splat(1.0e-12));

    const zero: @Vector(3, T) = @as(@Vector(3, T), @splat(0));
    const one: @Vector(3, T) = @as(@Vector(3, T), @splat(1));

    // Find vectors for two edges sharing v0
    const e1 = v1 - v0;
    const e2 = v2 - v0;

    // Begin calculating determinant - also used to calculate u parameter
    const p = zml.vec.cross(direction, e2);

    // If determinant is near zero, ray lies in plane of triangle
    var det = @as(@Vector(3, T), @splat(zml.vec.dot(e1, p)));

    // Check if determinant is near zero
    const det_near_zero = @abs(det) < epsilon;

    // when the determinant is near zero, return no intersection
    det = @select(T, det_near_zero, one, det);

    // Calculate distance from v0 to ray origin
    const s = origin - v0;

    // Calculate u parameter and test bounds
    const u = @as(@Vector(3, T), @splat(zml.vec.dot(s, p))) / det;

    // Prepare to test v parameter
    const q = zml.vec.cross(s, e1);

    // Calculate v parameter and test bounds
    const v = @as(@Vector(3, T), @splat(zml.vec.dot(direction, q))) / det;

    // get intersection point
    const t = @as(@Vector(3, T), @splat(zml.vec.dot(e2, q))) / det;

    const no_intersection =
        (det_near_zero | (u < zero)) | ((v < zero) | ((u + v) > one)) | (t < zero);

    return @select(T, no_intersection, @as(@Vector(3, T), @splat(std.math.floatMax(T))), t)[0];
}

test ray_aabb {
    const aabb: zml.geom.AABB(f32) = .from_two_points(.{ -1, -1, -1 }, .{ 1, 1, 1 });
    inline for (0..3) |axis| {
        {
            // Ray starting in the center of the box, pointing high
            const origin = @Vector(3, f32){ 0, 0, 0 };
            var direction = @Vector(3, f32){ 0, 0, 0 };
            direction[axis] = 1;
            const fraction = ray_aabb(aabb, origin, .from_direction(direction));
            try std.testing.expectApproxEqRel(-1.0, fraction, 1.0e-6);
        }
    }
}

test ray_triangle {
    const v0 = @Vector(3, f32){ 0, 0, 0 };
    const v1 = @Vector(3, f32){ 1, 0, 0 };
    const v2 = @Vector(3, f32){ 0, 1, 0 };

    {
        // Ray starting above the triangle, pointing down
        const origin = @Vector(3, f32){ 0.25, 0.25, 1 };
        const direction = @Vector(3, f32){ 0, 0, -1 };
        const fraction = ray_triangle(f32, origin, direction, v0, v1, v2);
        try std.testing.expectApproxEqRel(1.0, fraction, 1.0e-6);
    }
    {
        // Ray starting below the triangle, pointing up
        const origin = @Vector(3, f32){ 0.25, 0.25, -1 };
        const direction = @Vector(3, f32){ 0, 0, 1 };
        const fraction = ray_triangle(f32, origin, direction, v0, v1, v2);
        try std.testing.expectApproxEqRel(1.0, fraction, 1.0e-6);
    }
    {
        // Ray starting to the side of the triangle pointing away
        const origin = @Vector(3, f32){ -1, -1, 0 };
        const direction = @Vector(3, f32){ -1, -1, 0 };
        const fraction = ray_triangle(f32, origin, direction, v0, v1, v2);
        try std.testing.expectEqual(std.math.floatMax(f32), fraction);
    }
    {
        // Ray starting to the side of the triangle pointing towards
        const origin = @Vector(3, f32){ -1, -1, 0 };
        const direction = @Vector(3, f32){ 1, 1, 0 };
        const fraction = ray_triangle(f32, origin, direction, v0, v1, v2);
        try std.testing.expectEqual(std.math.floatMax(f32), fraction);
    }
}

test ray_cylinder {
    const cylinder: zml.geom.Cylinder(f32) = .from_two_points_radius(.{ 0, 0, 0 }, .{ 0, 0, 2 }, 1.0);
    {
        // Ray starting outside the cylinder, pointing towards
        const origin = @Vector(3, f32){ 2, 0, 1 };
        const direction = @Vector(3, f32){ -1, 0, 0 };
        const fraction = ray_cylinder(cylinder, origin, direction);
        try std.testing.expectApproxEqRel(1.0, fraction, 1.0e-6);
    }
    {
        // Ray starting outside the cylinder, pointing away
        const origin = @Vector(3, f32){ 2, 0, 1 };
        const direction = @Vector(3, f32){ 1, 0, 0 };
        const fraction = ray_cylinder(cylinder, origin, direction);
        try std.testing.expectEqual(std.math.floatMax(f32), fraction);
    }
    {
        // Ray starting inside the cylinder
        const origin = @Vector(3, f32){ 0.5, 0, 1 };
        const direction = @Vector(3, f32){ 1, 0, 0 };
        const fraction = ray_cylinder(cylinder, origin, direction);
        try std.testing.expectApproxEqRel(0.0, fraction, 1.0e-6);
    }
    {
        // Ray entering through the top cap from above.
        const origin = @Vector(3, f32){ 0, 0, 5 };
        const direction = @Vector(3, f32){ 0, 0, -1 };
        const fraction = ray_cylinder(cylinder, origin, direction);
        try std.testing.expectApproxEqRel(3.0, fraction, 1.0e-6);
    }
}

test ray_sphere {
    const sphere: zml.geom.Sphere(f32) = .from_center_radius(.{ 0, 0, 0 }, 1.0);
    {
        // Ray starting outside, pointing towards the sphere: enters at the near surface.
        const fraction = ray_sphere(sphere, .{ -3, 0, 0 }, .{ 1, 0, 0 });
        try std.testing.expectApproxEqRel(2.0, fraction, 1.0e-6);
    }
    {
        // Ray starting outside, pointing away: no hit.
        const fraction = ray_sphere(sphere, .{ -3, 0, 0 }, .{ -1, 0, 0 });
        try std.testing.expectEqual(std.math.floatMax(f32), fraction);
    }
    {
        // Ray starting inside the sphere: fraction 0.
        const fraction = ray_sphere(sphere, .{ 0, 0, 0 }, .{ 1, 0, 0 });
        try std.testing.expectApproxEqRel(0.0, fraction, 1.0e-6);
    }
    {
        // Ray missing the sphere entirely (parallel, offset by more than the radius).
        const fraction = ray_sphere(sphere, .{ -3, 2, 0 }, .{ 1, 0, 0 });
        try std.testing.expectEqual(std.math.floatMax(f32), fraction);
    }
}

test ray_capsule {
    // Capsule along z from (0,0,-1) to (0,0,1), radius 1: total extent z in [-2, 2].
    const capsule: zml.geom.Capsule(f32) = .{ .hemisphere_centers = .{ .{ 0, 0, -1 }, .{ 0, 0, 1 } }, .radius = 1 };
    {
        // Hit the cylindrical body from the side.
        const fraction = ray_capsule(capsule, .{ 3, 0, 0 }, .{ -1, 0, 0 });
        try std.testing.expectApproxEqRel(2.0, fraction, 1.0e-6);
    }
    {
        // Hit the top hemisphere cap coming straight down the axis.
        const fraction = ray_capsule(capsule, .{ 0, 0, 5 }, .{ 0, 0, -1 });
        try std.testing.expectApproxEqRel(3.0, fraction, 1.0e-6);
    }
    {
        // Ray starting inside the capsule: fraction 0.
        const fraction = ray_capsule(capsule, .{ 0, 0, 0 }, .{ 1, 0, 0 });
        try std.testing.expectApproxEqRel(0.0, fraction, 1.0e-6);
    }
    {
        // Ray passing outside the rounded end (would miss, but a naive infinite cylinder would hit).
        const fraction = ray_capsule(capsule, .{ 3, 0, 1.9 }, .{ -1, 0, 0 });
        try std.testing.expectEqual(std.math.floatMax(f32), fraction);
    }
}

test "InvDirection" {
    const dir = @Vector(3, f32){ 1.0, 0.0, -1.0 };
    const invDir = InvDirection(f32).from_direction(dir);
    try std.testing.expectEqual(invDir.is_parallel, @Vector(3, bool){ false, true, false });
    try std.testing.expectApproxEqRel(invDir.inv_direction[0], 1.0, 1.0e-6);
    try std.testing.expectApproxEqRel(invDir.inv_direction[1], 0.0, 1.0e-6);
    try std.testing.expectApproxEqRel(invDir.inv_direction[2], -1.0, 1.0e-6);
}
