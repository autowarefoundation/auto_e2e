#include "navigation_rasterizer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr std::int32_t kMapChannels = 14;
constexpr std::int32_t kRouteChannels = 2;
constexpr std::int32_t kDirectionSin = 7;
constexpr std::int32_t kDirectionCos = 8;
constexpr std::int32_t kDirectionValid = 9;
constexpr std::int32_t kKnownMapArea = 10;
constexpr std::int32_t kRoadLevel = 11;
constexpr std::int32_t kRoadLevelValid = 12;
constexpr std::int32_t kLevelAmbiguity = 13;
constexpr double kEpsilon = 1e-12;

struct Point {
  double x;
  double y;
};

bool valid_geometry(const NavGeometry& geometry) {
  if (geometry.height_px <= 0 || geometry.width_px <= 0 ||
      !std::isfinite(geometry.meters_per_pixel) ||
      geometry.meters_per_pixel <= 0.0) {
    return false;
  }
  const double expected_x =
      geometry.height_px * geometry.meters_per_pixel;
  const double expected_y =
      geometry.width_px * geometry.meters_per_pixel;
  return std::abs((geometry.x_max_m - geometry.x_min_m) - expected_x) <
             1e-9 &&
         std::abs((geometry.y_max_m - geometry.y_min_m) - expected_y) <
             1e-9;
}

std::size_t index_chw(std::int32_t channel, std::int32_t row,
                      std::int32_t column, const NavGeometry& geometry) {
  return (static_cast<std::size_t>(channel) * geometry.height_px + row) *
             geometry.width_px +
         column;
}

Point map_to_ego(const Point& point, const NavPose& pose) {
  const double dx = point.x - pose.x_enu_m;
  const double dy = point.y - pose.y_enu_m;
  const double cosine = std::cos(pose.yaw_rad);
  const double sine = std::sin(pose.yaw_rad);
  return {
      cosine * dx + sine * dy,
      -sine * dx + cosine * dy,
  };
}

Point ego_to_map(const Point& point, const NavPose& pose) {
  const double cosine = std::cos(pose.yaw_rad);
  const double sine = std::sin(pose.yaw_rad);
  return {
      pose.x_enu_m + cosine * point.x - sine * point.y,
      pose.y_enu_m + sine * point.x + cosine * point.y,
  };
}

Point pixel_center(std::int32_t row, std::int32_t column,
                   const NavGeometry& geometry) {
  return {
      geometry.x_max_m -
          (static_cast<double>(row) + 0.5) * geometry.meters_per_pixel,
      geometry.y_max_m -
          (static_cast<double>(column) + 0.5) *
              geometry.meters_per_pixel,
  };
}

Point ego_to_pixel(const Point& point, const NavGeometry& geometry) {
  return {
      (geometry.x_max_m - point.x) / geometry.meters_per_pixel - 0.5,
      (geometry.y_max_m - point.y) / geometry.meters_per_pixel - 0.5,
  };
}

double point_segment_distance(const Point& point, const Point& start,
                              const Point& end) {
  const double dx = end.x - start.x;
  const double dy = end.y - start.y;
  const double length_squared = dx * dx + dy * dy;
  if (length_squared <= kEpsilon) {
    return std::hypot(point.x - start.x, point.y - start.y);
  }
  const double projection =
      ((point.x - start.x) * dx + (point.y - start.y) * dy) /
      length_squared;
  const double fraction = std::clamp(projection, 0.0, 1.0);
  const double nearest_x = start.x + fraction * dx;
  const double nearest_y = start.y + fraction * dy;
  return std::hypot(point.x - nearest_x, point.y - nearest_y);
}

bool point_in_polygon(const Point& point, const std::vector<Point>& polygon) {
  bool inside = false;
  std::size_t previous = polygon.size() - 1;
  for (std::size_t current = 0; current < polygon.size();
       previous = current++) {
    const Point& a = polygon[current];
    const Point& b = polygon[previous];
    const bool crosses = ((a.y > point.y) != (b.y > point.y));
    if (!crosses) {
      continue;
    }
    const double x_intersection =
        (b.x - a.x) * (point.y - a.y) / (b.y - a.y) + a.x;
    if (point.x < x_intersection) {
      inside = !inside;
    }
  }
  return inside;
}

std::array<std::int32_t, 4> pixel_bounds(
    const std::vector<Point>& points, double padding_m,
    const NavGeometry& geometry) {
  double min_row = std::numeric_limits<double>::infinity();
  double max_row = -std::numeric_limits<double>::infinity();
  double min_column = std::numeric_limits<double>::infinity();
  double max_column = -std::numeric_limits<double>::infinity();
  const double padding_px = padding_m / geometry.meters_per_pixel + 1.0;
  for (const Point& point : points) {
    const Point pixel = ego_to_pixel(point, geometry);
    min_row = std::min(min_row, pixel.x - padding_px);
    max_row = std::max(max_row, pixel.x + padding_px);
    min_column = std::min(min_column, pixel.y - padding_px);
    max_column = std::max(max_column, pixel.y + padding_px);
  }
  return {
      std::max<std::int32_t>(
          0, static_cast<std::int32_t>(std::floor(min_row))),
      std::min<std::int32_t>(
          geometry.height_px - 1,
          static_cast<std::int32_t>(std::ceil(max_row))),
      std::max<std::int32_t>(
          0, static_cast<std::int32_t>(std::floor(min_column))),
      std::min<std::int32_t>(
          geometry.width_px - 1,
          static_cast<std::int32_t>(std::ceil(max_column))),
  };
}

float encode_level(std::int32_t level) {
  const float clamped =
      std::clamp(static_cast<float>(level), -8.0F, 8.0F);
  return (clamped + 8.0F) / 16.0F;
}

std::int32_t decode_level(float value) {
  return static_cast<std::int32_t>(
      std::lround(value * 16.0F - 8.0F));
}

bool apply_level(std::int32_t row, std::int32_t column,
                 const NavPrimitive& primitive, std::int32_t active_level,
                 bool active_level_valid, float* map_output,
                 const NavGeometry& geometry) {
  if (!primitive.level_valid) {
    return true;
  }
  const std::size_t valid_index =
      index_chw(kRoadLevelValid, row, column, geometry);
  const std::size_t level_index =
      index_chw(kRoadLevel, row, column, geometry);
  const std::size_t ambiguity_index =
      index_chw(kLevelAmbiguity, row, column, geometry);
  if (map_output[valid_index] > 0.5F &&
      decode_level(map_output[level_index]) != primitive.level) {
    map_output[ambiguity_index] = 1.0F;
  }
  if (active_level_valid && primitive.level != active_level) {
    map_output[ambiguity_index] = 1.0F;
    return false;
  }
  map_output[level_index] = encode_level(primitive.level);
  map_output[valid_index] = 1.0F;
  return true;
}

void write_semantic_pixel(std::int32_t row, std::int32_t column,
                          const NavPrimitive& primitive,
                          std::int32_t active_level, bool active_level_valid,
                          float* map_output,
                          const NavGeometry& geometry) {
  if (primitive.channel < 0 || primitive.channel >= kMapChannels) {
    return;
  }
  if (!apply_level(row, column, primitive, active_level,
                   active_level_valid, map_output, geometry)) {
    return;
  }
  map_output[index_chw(primitive.channel, row, column, geometry)] =
      primitive.value;
}

void draw_line(const std::vector<Point>& points,
               const NavPrimitive& primitive, std::int32_t active_level,
               bool active_level_valid, float* map_output,
               const NavGeometry& geometry, bool direction) {
  const double half_width =
      std::max(primitive.width_m * 0.5F,
               static_cast<float>(geometry.meters_per_pixel * 0.5));
  for (std::size_t segment = 0; segment + 1 < points.size(); ++segment) {
    const Point& start = points[segment];
    const Point& end = points[segment + 1];
    const std::vector<Point> segment_points{start, end};
    const auto bounds = pixel_bounds(segment_points, half_width, geometry);
    if (bounds[0] > bounds[1] || bounds[2] > bounds[3]) {
      continue;
    }
    const double theta = std::atan2(end.y - start.y, end.x - start.x);
    const float direction_sin =
        static_cast<float>((std::sin(theta) + 1.0) * 0.5);
    const float direction_cos =
        static_cast<float>((std::cos(theta) + 1.0) * 0.5);
    for (std::int32_t row = bounds[0]; row <= bounds[1]; ++row) {
      for (std::int32_t column = bounds[2]; column <= bounds[3];
           ++column) {
        const Point center = pixel_center(row, column, geometry);
        if (point_segment_distance(center, start, end) > half_width) {
          continue;
        }
        if (direction) {
          if (!apply_level(row, column, primitive, active_level,
                           active_level_valid, map_output, geometry)) {
            continue;
          }
          map_output[index_chw(kDirectionSin, row, column, geometry)] =
              direction_sin;
          map_output[index_chw(kDirectionCos, row, column, geometry)] =
              direction_cos;
          map_output[index_chw(kDirectionValid, row, column, geometry)] =
              1.0F;
        } else {
          write_semantic_pixel(row, column, primitive, active_level,
                               active_level_valid, map_output, geometry);
        }
      }
    }
  }
}

void draw_polygon(const std::vector<Point>& points,
                  const NavPrimitive& primitive,
                  std::int32_t active_level, bool active_level_valid,
                  float* map_output, const NavGeometry& geometry) {
  const auto bounds = pixel_bounds(points, 0.0, geometry);
  if (bounds[0] > bounds[1] || bounds[2] > bounds[3]) {
    return;
  }
  for (std::int32_t row = bounds[0]; row <= bounds[1]; ++row) {
    for (std::int32_t column = bounds[2]; column <= bounds[3];
         ++column) {
      if (point_in_polygon(pixel_center(row, column, geometry), points)) {
        write_semantic_pixel(row, column, primitive, active_level,
                             active_level_valid, map_output, geometry);
      }
    }
  }
}

void draw_point(const Point& point, const NavPrimitive& primitive,
                std::int32_t active_level, bool active_level_valid,
                float* map_output, const NavGeometry& geometry) {
  const double radius =
      std::max(primitive.width_m * 0.5F,
               static_cast<float>(geometry.meters_per_pixel * 0.5));
  const std::vector<Point> points{point};
  const auto bounds = pixel_bounds(points, radius, geometry);
  if (bounds[0] > bounds[1] || bounds[2] > bounds[3]) {
    return;
  }
  for (std::int32_t row = bounds[0]; row <= bounds[1]; ++row) {
    for (std::int32_t column = bounds[2]; column <= bounds[3];
         ++column) {
      const Point center = pixel_center(row, column, geometry);
      if (std::hypot(center.x - point.x, center.y - point.y) <= radius) {
        write_semantic_pixel(row, column, primitive, active_level,
                             active_level_valid, map_output, geometry);
      }
    }
  }
}

void draw_route_line(const std::vector<Point>& points, double corridor_width_m,
                     double rear_clip_m, std::uint8_t* route_output,
                     const NavGeometry& geometry) {
  const double half_width = corridor_width_m * 0.5;
  for (std::size_t segment = 0; segment + 1 < points.size(); ++segment) {
    const Point& start = points[segment];
    const Point& end = points[segment + 1];
    const std::vector<Point> segment_points{start, end};
    const auto bounds = pixel_bounds(segment_points, half_width, geometry);
    if (bounds[0] > bounds[1] || bounds[2] > bounds[3]) {
      continue;
    }
    for (std::int32_t row = bounds[0]; row <= bounds[1]; ++row) {
      for (std::int32_t column = bounds[2]; column <= bounds[3];
           ++column) {
        const Point center = pixel_center(row, column, geometry);
        if (center.x < -rear_clip_m) {
          continue;
        }
        if (point_segment_distance(center, start, end) <= half_width) {
          route_output[index_chw(0, row, column, geometry)] = 1;
        }
      }
    }
  }
}

void draw_destination(const Point& point, double radius_m,
                      std::uint8_t* route_output,
                      const NavGeometry& geometry) {
  if (point.x < geometry.x_min_m || point.x > geometry.x_max_m ||
      point.y < geometry.y_min_m || point.y > geometry.y_max_m) {
    return;
  }
  const std::vector<Point> points{point};
  const auto bounds = pixel_bounds(points, radius_m, geometry);
  for (std::int32_t row = bounds[0]; row <= bounds[1]; ++row) {
    for (std::int32_t column = bounds[2]; column <= bounds[3];
         ++column) {
      const Point center = pixel_center(row, column, geometry);
      if (std::hypot(center.x - point.x, center.y - point.y) <= radius_m) {
        route_output[index_chw(1, row, column, geometry)] = 1;
      }
    }
  }
}

bool valid_point_range(std::int32_t offset, std::int32_t count,
                       std::int32_t total) {
  return offset >= 0 && count >= 0 && offset <= total &&
         count <= total - offset;
}

bool inside(std::int32_t row, std::int32_t column,
            const NavGeometry& geometry) {
  return row >= 0 && row < geometry.height_px && column >= 0 &&
         column < geometry.width_px;
}

float bilinear_channel(const float* input, std::int32_t channel,
                       double row, double column,
                       const NavGeometry& geometry,
                       std::int32_t validity_channel,
                       double* total_weight) {
  const std::int32_t row0 =
      static_cast<std::int32_t>(std::floor(row));
  const std::int32_t column0 =
      static_cast<std::int32_t>(std::floor(column));
  const double row_fraction = row - row0;
  const double column_fraction = column - column0;
  float value = 0.0F;
  *total_weight = 0.0;
  for (std::int32_t row_offset = 0; row_offset <= 1; ++row_offset) {
    for (std::int32_t column_offset = 0; column_offset <= 1;
         ++column_offset) {
      const std::int32_t source_row = row0 + row_offset;
      const std::int32_t source_column = column0 + column_offset;
      if (!inside(source_row, source_column, geometry)) {
        continue;
      }
      const double row_weight =
          row_offset ? row_fraction : 1.0 - row_fraction;
      const double column_weight =
          column_offset ? column_fraction : 1.0 - column_fraction;
      const double weight = row_weight * column_weight;
      if (validity_channel >= 0 &&
          input[index_chw(validity_channel, source_row, source_column,
                          geometry)] <= 0.5F) {
        continue;
      }
      value += static_cast<float>(
          weight *
          input[index_chw(channel, source_row, source_column, geometry)]);
      *total_weight += weight;
    }
  }
  if (*total_weight <= kEpsilon) {
    return 0.0F;
  }
  return static_cast<float>(value / *total_weight);
}

}  // namespace

extern "C" {

const char* nav_renderer_version() {
  return "navigation_rasterizer_v1";
}

std::int32_t nav_render(
    const double* map_points_xy, std::int32_t map_point_count,
    const NavPrimitive* primitives, std::int32_t primitive_count,
    const double* route_points_xy, std::int32_t route_point_count,
    const std::int32_t* route_offsets, std::int32_t route_line_count,
    const double* destination_xy, std::int32_t destination_valid,
    NavPose ego_pose, NavGeometry geometry, float route_corridor_width_m,
    float destination_radius_m, float route_rear_clip_m,
    std::int32_t active_level, std::int32_t active_level_valid,
    float* map_output_chw, std::uint8_t* route_output_chw) {
  if (!valid_geometry(geometry) || map_point_count < 0 ||
      primitive_count < 0 || route_point_count < 0 ||
      route_line_count < 0 || map_output_chw == nullptr ||
      route_output_chw == nullptr ||
      (map_point_count > 0 && map_points_xy == nullptr) ||
      (primitive_count > 0 && primitives == nullptr) ||
      (route_point_count > 0 && route_points_xy == nullptr) ||
      (route_line_count > 0 && route_offsets == nullptr) ||
      (destination_valid && destination_xy == nullptr) ||
      route_corridor_width_m <= 0.0F || destination_radius_m <= 0.0F ||
      route_rear_clip_m <= 0.0F) {
    return 1;
  }
  const std::size_t pixels =
      static_cast<std::size_t>(geometry.height_px) * geometry.width_px;
  std::memset(map_output_chw, 0,
              pixels * kMapChannels * sizeof(float));
  std::memset(route_output_chw, 0,
              pixels * kRouteChannels * sizeof(std::uint8_t));

  for (std::int32_t primitive_index = 0;
       primitive_index < primitive_count; ++primitive_index) {
    const NavPrimitive& primitive = primitives[primitive_index];
    if (!valid_point_range(primitive.point_offset, primitive.point_count,
                           map_point_count) ||
        !std::isfinite(primitive.width_m) ||
        !std::isfinite(primitive.value)) {
      return 2;
    }
    std::vector<Point> points;
    points.reserve(primitive.point_count);
    for (std::int32_t point_index = 0;
         point_index < primitive.point_count; ++point_index) {
      const std::int32_t index = primitive.point_offset + point_index;
      const Point point{
          map_points_xy[2 * index],
          map_points_xy[2 * index + 1],
      };
      if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
        return 3;
      }
      points.push_back(map_to_ego(point, ego_pose));
    }
    switch (primitive.kind) {
      case NAV_LINE:
        if (points.size() >= 2) {
          draw_line(points, primitive, active_level,
                    active_level_valid != 0, map_output_chw, geometry, false);
        }
        break;
      case NAV_POLYGON:
        if (points.size() >= 3) {
          draw_polygon(points, primitive, active_level,
                       active_level_valid != 0, map_output_chw, geometry);
        }
        break;
      case NAV_POINT:
        if (!points.empty()) {
          draw_point(points.front(), primitive, active_level,
                     active_level_valid != 0, map_output_chw, geometry);
        }
        break;
      case NAV_DIRECTION_LINE:
        if (points.size() >= 2) {
          draw_line(points, primitive, active_level,
                    active_level_valid != 0, map_output_chw, geometry, true);
        }
        break;
      default:
        return 4;
    }
  }

  if (route_line_count > 0) {
    if (route_offsets[0] != 0 ||
        route_offsets[route_line_count] != route_point_count) {
      return 5;
    }
    for (std::int32_t line = 0; line < route_line_count; ++line) {
      const std::int32_t offset = route_offsets[line];
      const std::int32_t end = route_offsets[line + 1];
      if (!valid_point_range(offset, end - offset, route_point_count)) {
        return 6;
      }
      std::vector<Point> points;
      points.reserve(end - offset);
      for (std::int32_t index = offset; index < end; ++index) {
        const Point map_point{
            route_points_xy[2 * index],
            route_points_xy[2 * index + 1],
        };
        if (!std::isfinite(map_point.x) || !std::isfinite(map_point.y)) {
          return 7;
        }
        points.push_back(map_to_ego(map_point, ego_pose));
      }
      if (points.size() >= 2) {
        draw_route_line(points, route_corridor_width_m,
                        route_rear_clip_m, route_output_chw, geometry);
      }
    }
  }
  if (destination_valid) {
    const Point destination{
        destination_xy[0],
        destination_xy[1],
    };
    if (!std::isfinite(destination.x) || !std::isfinite(destination.y)) {
      return 8;
    }
    draw_destination(map_to_ego(destination, ego_pose),
                     destination_radius_m, route_output_chw, geometry);
  }
  return 0;
}

std::int32_t nav_warp(
    const float* map_input_chw, const std::uint8_t* route_input_chw,
    NavPose render_pose, NavPose sample_pose, NavGeometry geometry,
    float* map_output_chw, std::uint8_t* route_output_chw) {
  if (!valid_geometry(geometry) || map_input_chw == nullptr ||
      route_input_chw == nullptr || map_output_chw == nullptr ||
      route_output_chw == nullptr) {
    return 1;
  }
  const std::size_t pixels =
      static_cast<std::size_t>(geometry.height_px) * geometry.width_px;
  std::memset(map_output_chw, 0,
              pixels * kMapChannels * sizeof(float));
  std::memset(route_output_chw, 0,
              pixels * kRouteChannels * sizeof(std::uint8_t));

  const double rotation = render_pose.yaw_rad - sample_pose.yaw_rad;
  const double rotation_cos = std::cos(rotation);
  const double rotation_sin = std::sin(rotation);

  for (std::int32_t row = 0; row < geometry.height_px; ++row) {
    for (std::int32_t column = 0; column < geometry.width_px;
         ++column) {
      const Point sample_ego = pixel_center(row, column, geometry);
      const Point map_point = ego_to_map(sample_ego, sample_pose);
      const Point render_ego = map_to_ego(map_point, render_pose);
      const Point source_pixel = ego_to_pixel(render_ego, geometry);
      const std::int32_t nearest_row =
          static_cast<std::int32_t>(std::floor(source_pixel.x + 0.5));
      const std::int32_t nearest_column =
          static_cast<std::int32_t>(std::floor(source_pixel.y + 0.5));
      if (!inside(nearest_row, nearest_column, geometry)) {
        continue;
      }

      for (std::int32_t channel = 0; channel < kMapChannels;
           ++channel) {
        if (channel == kDirectionSin || channel == kDirectionCos ||
            channel == kDirectionValid || channel == kRoadLevel ||
            channel == kRoadLevelValid) {
          continue;
        }
        map_output_chw[index_chw(channel, row, column, geometry)] =
            map_input_chw[index_chw(channel, nearest_row, nearest_column,
                                    geometry)];
      }
      for (std::int32_t channel = 0; channel < kRouteChannels;
           ++channel) {
        route_output_chw[index_chw(channel, row, column, geometry)] =
            route_input_chw[index_chw(channel, nearest_row, nearest_column,
                                      geometry)];
      }

      if (map_output_chw[index_chw(kKnownMapArea, row, column, geometry)] <=
          0.5F) {
        continue;
      }

      double sin_weight = 0.0;
      double cos_weight = 0.0;
      const float encoded_sin = bilinear_channel(
          map_input_chw, kDirectionSin, source_pixel.x, source_pixel.y,
          geometry, kDirectionValid, &sin_weight);
      const float encoded_cos = bilinear_channel(
          map_input_chw, kDirectionCos, source_pixel.x, source_pixel.y,
          geometry, kDirectionValid, &cos_weight);
      if (sin_weight > kEpsilon && cos_weight > kEpsilon) {
        const double source_sin = 2.0 * encoded_sin - 1.0;
        const double source_cos = 2.0 * encoded_cos - 1.0;
        double sample_cos =
            rotation_cos * source_cos - rotation_sin * source_sin;
        double sample_sin =
            rotation_sin * source_cos + rotation_cos * source_sin;
        const double norm = std::hypot(sample_cos, sample_sin);
        if (norm > kEpsilon) {
          sample_cos /= norm;
          sample_sin /= norm;
          map_output_chw[
              index_chw(kDirectionSin, row, column, geometry)] =
              static_cast<float>((sample_sin + 1.0) * 0.5);
          map_output_chw[
              index_chw(kDirectionCos, row, column, geometry)] =
              static_cast<float>((sample_cos + 1.0) * 0.5);
          map_output_chw[
              index_chw(kDirectionValid, row, column, geometry)] = 1.0F;
        }
      }

      double level_weight = 0.0;
      const float level = bilinear_channel(
          map_input_chw, kRoadLevel, source_pixel.x, source_pixel.y,
          geometry, kRoadLevelValid, &level_weight);
      if (level_weight > kEpsilon) {
        map_output_chw[index_chw(kRoadLevel, row, column, geometry)] =
            level;
        map_output_chw[
            index_chw(kRoadLevelValid, row, column, geometry)] = 1.0F;
      }
    }
  }
  return 0;
}

}  // extern "C"
