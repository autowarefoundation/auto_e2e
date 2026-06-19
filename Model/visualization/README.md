# Trajectory Visualization #

This is implemented as a class. It contains 4 functions:
* `accel_and_curv_to_meters_trajectory`
* `meters_to_pixels_trajectory`
* `overlay_the_trajectory_with_map`
* `render_trajectory_map_tile`

## Complete function ##

### `render_trajectory_map_tile`

Integrates predicted trajectory into metric coordinates and 
draws them onto the raw BEV map tile.

It takes four inputs:
* `action_sequence`: (128, ) flattened (64, 2) $[acceleration, curvature]$ tensor.
It is the exact format of the trajectory outputted by `model()` function.
* `current_speed`: Scalar float from the egomotion history. 
This should be extracted from the egomotion history
* `map_image`: A map tile, not normalized. 
Ideally, it should follow L2D format. 
If the dataset does not provide maps directly, but provides GPS history, 
please use already existing [map generation function](https://github.com/autowarefoundation/auto_e2e/tree/main/Model/data_parsing/map_rendering)
to get a map tile.
* `radius_m`: The metric boundary of the `map_image` in meters.

Returns:
* A new PIL Image with the trajectory drawn on it.

## Helper functions ##

### `accel_and_curv_to_meters_trajectory`
Takes an action sequence containing pairs of acceleration ($m/s^2$) and curvature ($rad/m$), the ego vehicle's current speed, and a defined number of future timesteps. 
It integrates these values over time using a kinematic model to produce a tensor of metric $(X, Y)$ coordinates. The ego vehicle is positioned at the origin $(0.0, 0.0)$, with forward movement mapped to the positive $Y$-axis.

### `meters_to_pixels_trajectory`
Converts the metric trajectory tensor into 2D image pixel coordinates $(U, V)$. It scales the coordinates based on the pixel dimensions of the provided map image and the metric boundary `radius_m`.

### `overlay_the_trajectory_with_map`
Uses PIL's `ImageDraw` module to render the trajectory as a continuous green line directly onto a copy of the BEV map image. It also draws a red circle at the trajectory's origin to indicate the ego vehicle's current position.

## Dependencies ##

Core visualization logic only requires standard machine learning and image processing libraries:

```text
Pillow>=5.3.0
```

If you wish to run the live visualization script (`--live`) to test predictions using real dataset records, you will additionally need the [L2D dependencies](https://github.com/autowarefoundation/auto_e2e/tree/main/Model/data_parsing/l2d) 

## TODO ##

A function to draw the trajectory on the camera view is yet to be added.