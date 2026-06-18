# Nvidia Physical AI Data

## Data format description

```
data/
    camera/
        camera_front_wide_120fov/
            <clip_uuid>.camera_front_wide_120fov.mp4
            <clip_uuid>.camera_front_wide_120fov.timestamps.parquet
        camera_front_tele_30fov/    ...
        camera_cross_left_120fov/   ...
        camera_cross_right_120fov/  ...
        camera_rear_left_70fov/     ...
        camera_rear_right_70fov/    ...
        camera_rear_tele_30fov/     ...
    labels/
        egomotion/
            <clip_uuid>.egomotion.parquet
```

Each clip is 20s. The egomotion parquet contains 100Hz motion data; camera videos run at 30fps. Both use a shared per-clip timestamp reference (t=0 = clip anchor), which is used to align camera frames to egomotion moments.