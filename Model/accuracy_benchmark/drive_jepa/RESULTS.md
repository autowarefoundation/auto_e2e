# Drive-JEPA KITScenes validation comparison

All nine conditions completed on the identical 117 official validation scenes and 11,035 samples. IDs, timestamps, target arrays and selected metrics were independently checked from retained predictions.

Published sorted-UID SHA256 (terminal LF): `00f824202c97b47ec74561d247e8e30686e5505eef2ad2bc0acba97e41ce9925`.

## Selected trajectory ADE / FDE (metres)

| Model | Commands | 1 s | 2 s | 3 s | 4 s | 5 s |
|---|---|---:|---:|---:|---:|---:|
| Perception-free v1 | Missing | 0.205 / 0.420 | 0.510 / 1.191 | 0.934 / 2.328 | 1.482 / 3.829 | N/A |
| Perception-free v1 | 20 m | 0.221 / 0.445 | 0.533 / 1.222 | 0.961 / 2.362 | 1.511 / 3.864 | N/A |
| Perception-free v1 | 100 m | 0.222 / 0.445 | 0.533 / 1.223 | 0.961 / 2.365 | 1.512 / 3.868 | N/A |
| Perception-based v1 | Missing | 0.436 / 0.850 | 0.921 / 1.922 | 1.516 / 3.401 | 2.243 / 5.340 | N/A |
| Perception-based v1 | 20 m | 0.447 / 0.829 | 0.934 / 1.965 | 1.541 / 3.450 | 2.276 / 5.386 | N/A |
| Perception-based v1 | 100 m | 0.446 / 0.826 | 0.931 / 1.957 | 1.536 / 3.438 | 2.268 / 5.367 | N/A |
| Perception-based v2 | Missing | 0.670 / 1.270 | 1.371 / 2.773 | 2.163 / 4.643 | 3.067 / 6.770 | N/A |
| Perception-based v2 | 20 m | 0.715 / 1.345 | 1.455 / 2.917 | 2.266 / 4.774 | 3.177 / 6.880 | N/A |
| Perception-based v2 | 100 m | 0.715 / 1.344 | 1.455 / 2.922 | 2.268 / 4.786 | 3.182 / 6.898 | N/A |

## Proposal metrics (metres)

For perception-based checkpoints, these are separate from the selected trajectory. Oracle minima use targets offline.

| Model | Commands | Policy | 1 s ADE/FDE | 2 s ADE/FDE | 3 s ADE/FDE | 4 s ADE/FDE | 5 s |
|---|---|---|---:|---:|---:|---:|---:|
| Perception-based v1 | Missing | candidate_first | 0.674 / 1.228 | 1.275 / 2.471 | 1.952 / 4.066 | 2.784 / 6.267 | N/A |
| Perception-based v1 | Missing | candidate_mean | 0.842 / 1.574 | 1.610 / 3.087 | 2.454 / 5.044 | 3.434 / 7.512 | N/A |
| Perception-based v1 | Missing | oracle_min | 0.113 / 0.190 | 0.246 / 0.407 | 0.407 / 0.685 | 0.615 / 1.107 | N/A |
| Perception-based v1 | 20 m | candidate_first | 0.620 / 1.120 | 1.174 / 2.292 | 1.806 / 3.769 | 2.565 / 5.758 | N/A |
| Perception-based v1 | 20 m | candidate_mean | 0.819 / 1.504 | 1.553 / 2.975 | 2.361 / 4.827 | 3.289 / 7.155 | N/A |
| Perception-based v1 | 20 m | oracle_min | 0.114 / 0.180 | 0.240 / 0.393 | 0.397 / 0.660 | 0.599 / 1.038 | N/A |
| Perception-based v1 | 100 m | candidate_first | 0.615 / 1.111 | 1.167 / 2.279 | 1.796 / 3.750 | 2.553 / 5.744 | N/A |
| Perception-based v1 | 100 m | candidate_mean | 0.817 / 1.499 | 1.548 / 2.967 | 2.354 / 4.820 | 3.283 / 7.149 | N/A |
| Perception-based v1 | 100 m | oracle_min | 0.114 / 0.179 | 0.240 / 0.392 | 0.396 / 0.660 | 0.599 / 1.042 | N/A |
| Perception-based v2 | Missing | candidate_first | 0.372 / 0.751 | 0.826 / 1.766 | 1.400 / 3.252 | 2.109 / 5.089 | N/A |
| Perception-based v2 | Missing | candidate_mean | 0.732 / 1.330 | 1.429 / 2.823 | 2.222 / 4.658 | 3.106 / 6.691 | N/A |
| Perception-based v2 | Missing | oracle_min | 0.133 / 0.211 | 0.263 / 0.424 | 0.423 / 0.711 | 0.632 / 1.166 | N/A |
| Perception-based v2 | 20 m | candidate_first | 0.369 / 0.743 | 0.838 / 1.769 | 1.383 / 3.120 | 2.061 / 4.980 | N/A |
| Perception-based v2 | 20 m | candidate_mean | 0.747 / 1.364 | 1.462 / 2.876 | 2.253 / 4.662 | 3.128 / 6.680 | N/A |
| Perception-based v2 | 20 m | oracle_min | 0.135 / 0.211 | 0.264 / 0.425 | 0.426 / 0.712 | 0.637 / 1.181 | N/A |
| Perception-based v2 | 100 m | candidate_first | 0.363 / 0.730 | 0.827 / 1.751 | 1.370 / 3.103 | 2.047 / 4.961 | N/A |
| Perception-based v2 | 100 m | candidate_mean | 0.744 / 1.360 | 1.457 / 2.868 | 2.247 / 4.655 | 3.123 / 6.675 | N/A |
| Perception-based v2 | 100 m | oracle_min | 0.134 / 0.210 | 0.263 / 0.427 | 0.425 / 0.719 | 0.637 / 1.195 | N/A |

Perception-free runs retain one output trajectory; first/mean/min over a one-element candidate set would coincide with its selected result.

## Provenance and limitations

- Missing commands use an all-zero four-dimensional vector; invalid supplied routes use the distinct unknown one-hot category.
- 20 m commands use centreline arc length and +/-2 m lateral thresholds from the KITScenes recorded ego-pose origin. No NAVSIM vehicle-specific rear-axle offset is fabricated.
- 100 m commands are published AutoE2E maneuver labels. Routes are reconstructed offline from whole recorded scene traces; exact published navigation hashes match. Commands do not directly inspect future evaluation targets, but these are offline reconstructed routes rather than independently supplied online navigation plans.
- KITScenes raw front images and actual calibration are used. Native image preprocessing crops 28 pixels top/bottom, resizes to 512x256 and uses current and 0.5-second-past frames. Perception-based projection retains native 0.4 scale and 32-pixel padding with actual front shape and calibration; an unused metadata slot aliases the front camera to accommodate KITScenes unequal camera resolutions.
- Causal KITScenes ego velocity/acceleration and the recorded ego origin are used; trained model vehicle geometry is retained. This is a dataset adaptation.
- NAVSIM v2 selection uses its native learned-score branch without optional sequential simulator comfort calibration. This is independent-sample KITScenes ADE/FDE, not native NAVSIM sequential simulator scoring.
- Perception-free inference used bf16; perception-based inference used fp32. Checkpoints were strictly loaded, and trained model forward computation was retained.
- ADE averages displacement error over 10Hz samples through each horizon; FDE uses its endpoint. Native 0.5-second predictions are interpolated from the origin without extrapolation. Metrics weight each sample equally.
- Candidate mean averages errors across all 32 proposals per sample. Oracle minima choose ADE and FDE independently per sample and horizon using ground truth offline; these are not deployed selection metrics. Candidate first means proposal index zero, not learned-score selection.
- 5-second ADE/FDE are unavailable: these checkpoints predict only 4 seconds.

20 m command counts: 827 left, 844 right, 5,914 straight, 3,450 unknown (2,980 invalid route/map, 260 route gaps, 210 short routes).

100 m command counts: 920 left, 800 right, 6,335 straight, 2,980 unknown.

The uploaded 140-episode mirror includes 117 official val scenes and 23 overlap_train_val scenes. The latter were excluded to match the published 11,035-sample validation population.

Exact values, checkpoint hashes, run provenance and all per-horizon candidate metrics are retained in reference_results.json; scene predictions and individual reports remain in their original result directories.
