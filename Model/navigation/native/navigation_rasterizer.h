#pragma once

#include <cstdint>

#if defined(_WIN32)
#define NAV_API __declspec(dllexport)
#else
#define NAV_API __attribute__((visibility("default")))
#endif

extern "C" {

enum NavPrimitiveKind : std::int32_t {
  NAV_LINE = 0,
  NAV_POLYGON = 1,
  NAV_POINT = 2,
  NAV_DIRECTION_LINE = 3,
};

struct NavGeometry {
  std::int32_t height_px;
  std::int32_t width_px;
  double meters_per_pixel;
  double x_min_m;
  double x_max_m;
  double y_min_m;
  double y_max_m;
};

struct NavPose {
  double x_enu_m;
  double y_enu_m;
  double yaw_rad;
};

struct NavPrimitive {
  std::int32_t point_offset;
  std::int32_t point_count;
  std::int32_t kind;
  std::int32_t channel;
  std::int32_t level;
  std::int32_t level_valid;
  float width_m;
  float value;
};

NAV_API const char* nav_renderer_version();

NAV_API std::int32_t nav_render(
    const double* map_points_xy,
    std::int32_t map_point_count,
    const NavPrimitive* primitives,
    std::int32_t primitive_count,
    const double* route_points_xy,
    std::int32_t route_point_count,
    const std::int32_t* route_offsets,
    std::int32_t route_line_count,
    const double* destination_xy,
    std::int32_t destination_valid,
    NavPose ego_pose,
    NavGeometry geometry,
    float route_corridor_width_m,
    float destination_radius_m,
    float route_rear_clip_m,
    std::int32_t active_level,
    std::int32_t active_level_valid,
    float* map_output_chw,
    std::uint8_t* route_output_chw);

NAV_API std::int32_t nav_warp(
    const float* map_input_chw,
    const std::uint8_t* route_input_chw,
    NavPose render_pose,
    NavPose sample_pose,
    NavGeometry geometry,
    float* map_output_chw,
    std::uint8_t* route_output_chw);
}
