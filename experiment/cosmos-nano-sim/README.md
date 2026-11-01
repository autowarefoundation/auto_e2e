# Cosmos3-Nano-Sim-Auto on nuPlan: shifted-viewpoint camera generation

This experiment checks whether NVIDIA Cosmos3-Nano-Sim-Auto (the "Cosmos 3 Nano Sim"
model raised in #221) can generate the six AutoE2E cameras from an ego pose that is
displaced from the recorded drive. Two questions:

1. Could it produce lateral-recovery training data, instead of warping the recorded
   images (branch `feat/lateral-recovery-kl`)?
2. Could it replace NuRec as the AlpaSim renderer (Design/reinforcement_learning.md §4)?

Results are in [RESULTS.md](./RESULTS.md). Nothing under `Model/` or `Platform/` is
changed. The raw nuPlan data is only read.

## What the model needs

Sim-Auto does not warp images. It generates video from per-camera WSM control videos,
which are renderings of lane lines, road edges, crosswalks and object boxes along the
ego trajectory, plus one caption per camera and the first RGB frame of each camera.
Moving the ego therefore means rendering new WSM controls from a moved trajectory.

| Step | Script | Output |
| --- | --- | --- |
| 1 | `setup.sh` | uv, cosmos-framework `55f0554`, inference venv (Python 3.13, CUDA 13), WSM4 renderer venv, Sim-Auto weights `e289a50` (31 GB) |
| 2 | `nuplan_to_rdshq.py` | One 4 s nuPlan window as RDS-HQ, once per ego shift (0, 0.5 m, 1.0 m, 1.0 m + 3°), plus the undistorted recorded frames |
| 3 | `render_and_request.py` | WSM4 control videos for each shift, caption files and one Sim-Auto request per shift |
| 4 | `generate.sh` | Sim-Auto generation for a request file, with peak GPU memory and wall time |
| 5 | `compare.py` | Comparison grids, WSM/RGB blends and `metrics.json` |

`run.sh` runs steps 1 to 5. `sample_run.sh <views> <480|720> <frames>` runs NVIDIA's
shared sample with fewer views or frames, to check the GPU and time the model.

### Data

The scene is nuPlan v1.1 mini log `2021.05.12.22.28.35_veh-35_00620_01164` (Las Vegas
Strip), starting at the first window where the ego drives straight at 5 m/s or more.
The three archives (mini DB, mini camera 0, maps v1.1) are read with HTTP range requests
from the official nuPlan open-data bucket `s3://motional-nuplan/public/nuplan-v1.1/`.
Their size and ETag are checked against the `source_etag` recorded in the Platform raw
snapshot `nuplan-v1.1-mini-group0-20260809`, so the bytes are the snapshot's and no
Platform credentials are needed on the instance.

### Choices made to fit nuPlan to the renderer

- Camera keys: the model has a rig-view embedding per physical camera of NVIDIA's
  Hyperion rig. Each nuPlan camera takes the nearest key: `CAM_F0` front-wide 120°,
  `CAM_L0`/`CAM_R0` cross-left/right 120°, `CAM_L2`/`CAM_R2` rear-left/right 70°,
  `CAM_B0` rear-tele 30°. The real nuPlan field of view is about 64° for all six, so
  every view is off the reference rig; NVIDIA warns this degrades quality.
- Cameras are undistorted the same way as the AutoE2E nuPlan pack
  (`cv2.getOptimalNewCameraMatrix` with alpha 0). The renderer only draws boxes
  through its F-theta path, so each undistorted pinhole is written as an F-theta
  camera whose polynomial fits `f·tan θ` (worst error 0.07 px).
- nuPlan maps are 2D and have no paint classes: lane dividers are written as
  `WHITE/OTHER` lines, the outline of the drivable area as `road_boundary`, and all
  map vertices at the ground height under the vehicle boxes. Traffic lights, signs,
  wait lines and road markings are left empty.
- Ego poses are interpolated from the 100 Hz `ego_pose` table to 30 Hz. Boxes use the
  nearest 20 Hz lidar sweep for each frame.
- Shifts ramp in over 2 s with a smoothstep, so frame 0 matches the recorded image that
  conditions the model.

## Hardware

The rolling (causal) inference of Sim-Auto needs FlashAttention-4 on Hopper (sm90) or
datacenter Blackwell (sm100/sm103). It refuses other GPUs
(`cosmos_framework/model/generator/mot/maskless_attention.py`, "maskless replay has no
other same-view path"). No g-family instance qualifies: L4 and L40S are sm89, RTX PRO 6000
(g7e) is sm120. This experiment used one p5.4xlarge (1x H100 80GB).

## Reproduce

The instance runs in a public subnet of the default VPC of us-east-2, with a security
group that has no inbound rules; it is reached only through SSM. Tag everything with
`Purpose=experiment-cosmos-nano-sim`.

```bash
# Security group with no inbound rules (outbound is open by default).
SG=$(aws ec2 create-security-group --region us-east-2 --vpc-id <default-vpc-id> \
  --group-name cosmos-nano-sim-exp-sg --description "no inbound, SSM only" \
  --tag-specifications 'ResourceType=security-group,Tags=[{Key=Purpose,Value=experiment-cosmos-nano-sim}]' \
  --query GroupId --output text)

# One p5.4xlarge Spot instance with the Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.10 (Ubuntu 24.04).
aws ec2 run-instances --region us-east-2 --image-id <dlami-id> --instance-type p5.4xlarge \
  --subnet-id <default-subnet-id> --security-group-ids "${SG}" --associate-public-ip-address \
  --iam-instance-profile Name=<ssm-instance-profile> \
  --metadata-options HttpTokens=required,HttpEndpoint=enabled \
  --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}' \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=300,VolumeType=gp3,DeleteOnTermination=true}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Purpose,Value=experiment-cosmos-nano-sim}]' \
                       'ResourceType=volume,Tags=[{Key=Purpose,Value=experiment-cosmos-nano-sim}]'
```

Copy this directory to `/home/ubuntu/cosmos-nano-sim/` on the instance and, as `ubuntu`:

```bash
cd /home/ubuntu/cosmos-nano-sim
bash run.sh > run.log 2>&1   # about 20 minutes on a p5.4xlarge
```

Everything, including the 31 GB of weights, is written under `/home/ubuntu/cosmos-nano-sim`
on the EBS root volume (about 60 GB). Outputs are in `outputs/nuplan/` (generated and
control videos per shift) and `outputs/compare/`.

Delete the resources afterwards:

```bash
aws ec2 terminate-instances --region us-east-2 --instance-ids <instance-id>
aws ec2 delete-security-group --region us-east-2 --group-id "${SG}"
```
