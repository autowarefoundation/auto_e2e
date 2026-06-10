# AutoE2E Architecture

## Architecture Diagram
<img src="../Media/auto_e2e_architecture.jpg" width="100%">

## Inputs and Predictions
**AutoE2E consumes as input:**
- 7 camera images at 224x224 resolution (providing a surround view of the vehicle alongside a telephoto front and rear camera for long range viewing)
- Rendered map tile (indicating the high level road network layout and future route of the vehicle)
- Egomotion history (speed, acceleration, yaw angle and yaw angle rate for the previous 6.4s at 10Hz sampling rate)

**AutoE2E outputs a prediction of:**
- Future driving trajectory (modelled as future acceleration and curvature values over a 6.4s future horizon at 10Hz sampling rate)
- `ego_hidden` — 256-dim final GRU hidden state from `TrajectoryPlanner` summarising the planner's intent over the prediction horizon. Conditions `FutureState`; replaces the legacy compressed visual feature vector / rolling visual history buffer.

**During training, and for purposes of model introspection, AutoFSD also predicts:**
- Future visual features at 1.6s intervals for a 6.4s future horizon (what does the future feature representation of the scene look like, this is used for a feature reconstruction loss similar to JEPA)

**Forward signature:**
```python
trajectory, ego_hidden, future_visual_features = model(
    visual_tiles,        # (B, V, 3, H, W) — 7 cameras + 1 map tile
    egomotion_history,   # (B, 256)
    camera_params=None,  # (B, V, 3, 4) optional, used by BEV fusion
    mode="train",        # "train" returns future_visual_features; otherwise None
)
```

**To learn the driving policy:**
- Imitaiton Learning is used to penalize trajectory prediciton as well as World Model Simulation based Reinforcement Learning


