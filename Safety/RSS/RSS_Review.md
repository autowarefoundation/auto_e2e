---
title: Responsibility-Sensitive Safety and Mobileye's Safety Framework
subtitle: A Reference Review for the AutoE2E Safety Argument
---

# Responsibility-Sensitive Safety and Mobileye's Safety Framework

*A Reference Review for the AutoE2E Safety Argument — prepared for the Autoware Foundation
Robotaxi Working Group.* [AutoE2E_RSS_Safety_Review.docx](https://docs.google.com/document/d/1hexKVAJLx5BqKb5Rcu10RPiEzkM82XeZ/edit?usp=sharing&ouid=116038880172893009378&rtpof=true&sd=true)

> Working document. This review synthesizes publicly available Mobileye/Intel material to
> extract a reusable safety-argument structure and candidate performance goals for AutoE2E.
> It does not constitute a certification artifact and is not a substitute for a formal HARA
> or safety case.

## Table of contents

1. [Executive Summary](#1-executive-summary)
2. [Purpose and Scope](#2-purpose-and-scope)
3. [Why a Formal Safety Argument Is Needed](#3-why-a-formal-safety-argument-is-needed)
4. [Functional Safety vs. Nominal Safety](#4-functional-safety-vs-nominal-safety)
5. [The RSS Model: Five Rules and Formal Structure](#5-the-rss-model-five-rules-and-formal-structure)
6. [Validating Soundness: Mapping RSS to NHTSA Pre-Crash Scenarios](#6-validating-soundness)
7. [Standardization and Industry Adoption](#7-standardization-and-industry-adoption)
8. [The Broader Safety-by-Design Framework](#8-the-broader-safety-by-design-framework)
9. [RSS in Production: Mobileye Drive™ and SuperVision™](#9-rss-in-production)
10. [Translating This Into Candidate AutoE2E Goals](#10-translating-this-into-candidate-autoe2e-goals)
11. [AutoE2E Benchmark Plan](#11-autoe2e-benchmark-plan)
12. [Where to Solidify the Architecture for Safety](#12-where-to-solidify-the-architecture-for-safety)
13. [References](#13-references)
---

## 1. Executive Summary

This review constitute a deep dive into Mobileye's Drive™ and SuperVision™ systems and into the
Responsibility-Sensitive Safety (RSS) framework, with the goal of extracting a safety
argument structure AutoE2E can compare itself against and, from it, a set of concrete
performance goals. The purpose is explicitly to give the team a target to reach — a fixed reference point that lets benchmarking, milestone
planning, and the paper's safety framing all point at the same goalposts.

RSS, introduced by Shalev-Shwartz, Shammah and Shashua in 2017 and refined since, is a
white-box mathematical model that formalizes what it means for a vehicle to drive
responsibly: it defines when a situation is dangerous, what a proper response to that
danger looks like, and therefore who is to blame if a collision occurs. Its central claim
is that statistical mileage-based validation alone cannot scale to the reliability
autonomous vehicles need, so a model-based, verifiable notion of safety is required
alongside it.

Mobileye has since folded RSS into a broader safety-by-design framework — co-authored with
Aptiv, Audi, BMW, Baidu, Continental, Daimler, FCA, HERE, Infineon, Intel and Volkswagen —
that ties RSS together with ISO 21448 (SOTIF), ISO 26262 (functional safety) and automotive
cybersecurity, and operationalizes it as a set of fail-safe and fail-degraded system
capabilities. This same stack (RSS + REM crowdsourced mapping + True Redundancy sensing) is
what ships today inside Mobileye Drive™ (the L4 robotaxi platform) and SuperVision™ (the
L2+ consumer bridge system) — the closest commercial analogs to what the Robotaxi WG is
building.

## 2. Purpose and Scope

Scoped from two action items:
- [ ] a deep dive of Mobileye Drive™ and SuperVision™,
- [ ] a review of Mobileye's RSS paper to extract a safety argument and set a performance bar for AutoE2E.

Covers: (a) the RSS formal model and its safety-argument structure; (b) how RSS relates to
ISO 21448 (SOTIF) and ISO 26262; (c) how RSS is implemented in Drive™/SuperVision™; (d)
candidate goals and open questions for AutoE2E.

Out of scope: a full ASIL decomposition for AutoE2E (separate, ongoing, informed by the
VisionPilot HARA work), and a legal/certification opinion on RSS's regulatory status.

## 3. Why a Formal Safety Argument Is Needed

The RSS paper opens by arguing that two things stand between autonomous driving and mass
deployment: the lack of a standardized, verifiable definition of safety, and the lack of a
scalable way to validate it. The statistical argument against mileage-based validation alone:

Human driving has a fatality rate of roughly one per million hours. For an autonomous
system to be accepted as meaningfully safer — three orders of magnitude, i.e. one fatality
per billion hours — a purely statistical validation would require on the order of **30
billion miles** of testing [1], and that evidence would need to be regathered after every
software change, since a single altered line of planning code invalidates the prior
validation. RAND's independent 2016 analysis reaches a comparable conclusion via a
different method, estimating hundreds of millions to hundreds of billions of miles
depending on assumptions [7].

The other two commonly cited validation strategies — counting driver disengagements, and
validating in simulation — carry their own structural problems: disengagement counts
conflate "almost an accident" with "an accident" for rare, difficult cases, and any
simulator accurate enough to validate one driving policy isn't guaranteed to remain
accurate once the policy changes.

**Conclusion:** data and simulation must be paired with a white-box, interpretable model
that can be reasoned about and proven correct independently of mileage logged. RSS is
offered as that model.

## 4. Functional Safety vs. Nominal Safety

**Functional safety** (ISO 26262) concerns whether hardware/software operate without
fault. **Nominal safety** concerns whether the vehicle's decisions are safe assuming
everything works exactly as intended — a vehicle can be perfectly functionally safe and
still drive itself into an accident through bad decision logic.

The RSS paper's claim that no standard covered nominal safety was accurate in 2017 but is
dated: **ISO/PAS 21448 (SOTIF) was first published in 2019** specifically to close that
gap, and the "Safety First for Automated Driving" whitepaper [3] (also 2019) treats SOTIF
as the standard governing exactly this question (§8.2).

**SOTIF is a process standard** — how to develop/verify/validate so known-risky and
unknown behaviors are progressively identified and closed, without specifying what a
correct decision looks like. **RSS is a content model** — specific, checkable rules for
what a safe decision is. Complementary, not competing: RSS-style rules are one candidate
way to define the "safe behavior" a SOTIF process then verifies against.

For AutoE2E: a learned end-to-end planner's safety questions are almost entirely
nominal-safety questions (is the policy's logic sound), not functional-safety questions
(did hardware fail) — the latter belongs to the platform's ISO 26262 process (VisionPilot's
HARA).

## 5. The RSS Model: Five Rules and Formal Structure

RSS formalizes the legal *Duty of Care* concept. Three design goals: **sound** (matches
human judgment about blame), **useful** (assertive, not overly defensive), **efficiently
verifiable** (provable, not just plausible).

| Rule | Informal statement | Formal mechanism |
|---|---|---|
| 1. Safe distance | Don't hit the car in front of you. | Minimum longitudinal gap from both cars' velocities and worst-case braking/acceleration; falling under it starts a "dangerous situation." |
| 2. No reckless cut-ins | Don't cut into another car's safety envelope. | Minimum lateral gap, same derivation, respected on lane changes/merges. |
| 3. Right-of-way given, not taken | Priority doesn't excuse causing a collision. | The prioritized agent must still brake if the other can't safely stop. |
| 4. Caution under limited visibility | Be careful where you can't see. | "Exposure time" — first moment an object could be seen — agent must respond safely from then on. |
| 5. Avoid a crash if you can without causing another | If you can prevent an accident safely, you must. | A "legal evasive manoeuvre" is required whenever one exists that doesn't itself violate rules 1–4. |

Formal constructs: **dangerous situation** (safe longitudinal + lateral distances both
violated), **danger threshold time** (exact moment it became dangerous), **proper
response** (bounded required action once dangerous), **responsibility** (assigned to
whoever failed the proper response — provably, by induction: if every agent always
executes proper response, no attributable collision occurs — "Utopia is possible").

Proper responses are computed **pairwise** (one other agent at a time), proven never to
produce contradictory instructions even in dense traffic — decomposability into
independent pairwise constraints, a useful design lens regardless of adopting RSS's exact
formulas.

### 5.1 Occlusion and vulnerable road users (VRUs)

Occluded space is assumed occupied at a *bounded* "reasonable" worst-case speed until
exposure time (unbounded would force a near-stop near any parked car). For pedestrians/VRUs:
same safe-distance/proper-response logic reapplied with human-kinematics parameters
(~2 m/s² acceleration bound, directional uncertainty); priority set contextually
(pedestrians on residential streets, vehicles on fast roads, crossings follow their signal).

### 5.2 A semantic action space for planning

The paper's second half addresses making a driving policy computationally tractable: raw
geometric action spaces (specific accelerations/trajectories) are intractably large with
poor signal-to-noise for learning a Q-function. Their fix — a **semantic action space**
("follow the car ahead at two seconds," not raw control values) — keeps the space small
(tens of discrete goals) while supporting long horizons. A design pattern, not a safety
requirement, but relevant to AutoE2E's own planner/reasoning-band horizon-aware action
representations.

## 6. Validating Soundness: Mapping RSS to NHTSA Pre-Crash Scenarios {#6-validating-soundness}

Mobileye applied RSS to NHTSA's pre-crash scenario typology [2] — 37 categories from the
2004 GES crash database, ~99.4% of light-vehicle crashes. A reusable template: take an
external, independently-produced taxonomy of real crash patterns and show the safety
model's blame-assignment logic matches human judgment in each case.

| Scenario family | Representative cases | How RSS resolves it |
|---|---|---|
| One-way traffic | Lead vehicle braking/stopped | Rear car at fault if it fails the computed safe following distance |
| Cut-in / drifting | Lane change/drift into another's path | Fault depends on longitudinal safe-distance violation at the moment of the change; other car must still make minimal evasive effort |
| Two-way traffic | Overtake/drift into oncoming lane | Whichever car first violates the asymmetric opposite-direction safe distance is at fault |
| Multiple geometry / right-of-way | Running a light/stop sign, unprotected turn | "Priority given, not taken" — prioritized vehicle still expected to brake if it had time |
| VRUs and occlusion | Pedestrian/cyclist from behind an obstruction | Vehicle cleared only if not driving unreasonably fast for the occlusion and slowed monotonically once visible |

**Conclusion:** RSS's four-part structure (safe distance → dangerous situation → proper
response → responsibility) covers essentially all 37 NHTSA categories without
scenario-specific special cases. AutoE2E could reuse this as a validation methodology: a
small formal core, shown to hold against an external scenario library.

## 7. Standardization and Industry Adoption

| Body / standard | Status | Relationship to RSS |
|---|---|---|
| IEEE 2846 | In development | Open industry working group, using RSS as its starting reference model |
| China ITS Industry Alliance — 0116-2019 | Approved (v1.0) | National AV decision-making safety standard, built directly on RSS's structure |
| ISO 4804 | Referenced via [3] | Safety-by-design/V&V guidance for SAE L3/L4; whitepaper §9 operationalizes RSS-adjacent concepts against it |
| ISO 21448 (SOTIF) | Published | Same "nominal safety" gap RSS addresses; complementary, not competing |

> None of the above claims AutoE2E is or should be RSS-compliant — it's included so the WG
> can decide, with the landscape visible, whether to align to SOTIF alone, an RSS-style
> explicit rule layer, or both.

## 8. The Broader Safety-by-Design Framework

In 2019, Mobileye/Intel co-published "Safety First for Automated Driving" [3] with Aptiv,
Audi, BMW, Baidu, Continental, Daimler, FCA, HERE, Infineon and Volkswagen — guidance, not a
binding standard, but the most complete public description of turning RSS-adjacent thinking
into an actual system architecture.

### 8.1 Twelve principles

Organized around five themes: security; safe operation and degradation;
operational-design-domain (ODD) determination; handover between vehicle and operator;
behavior in traffic. Two map directly onto capabilities AutoE2E will need: detecting when
it's outside its supported operating envelope, and having a well-defined, minimal-risk way
to hand back control or stop.

### 8.2 SOTIF: the known/unknown risk model

Three-region model: **known-safe**, **known-but-potentially-unsafe**, **unknown**. Goals:
grow the known-safe region, shrink known-risky via verification (simulation, targeted
testing), shrink unknown via validation (endurance testing, field data).

> This maps closely onto the open/closed/gap-register structure already used in the
> VisionPilot HARA work — the SOTIF model is the same idea in more general form.

### 8.3 Functional safety and cybersecurity as sibling pillars

SOTIF, ISO 26262 functional safety, and cybersecurity as three interacting pillars: a
system can't be called safe unless also adequately secure, since an attacker can turn a
functionally sound system into an unsafe one.

### 8.4 Capabilities: fail-safe and fail-degraded

Seven fail-safe (FS) capabilities deliver the core driving function; six fail-degraded (FD)
capabilities keep the system in a tolerable-risk state when something goes wrong.

| ID | Capability | Description |
|---|---|---|
| FS_1 | Determine location | Know where the vehicle is relative to its ODD |
| FS_2 | Perceive relevant objects | Detect all static/dynamic entities that matter |
| FS_3 | Predict future behavior | Forecast movement, including occluded objects |
| FS_4 | Create a collision-free, lawful plan | Respect safe distances and traffic rules |
| FS_5 | Execute the plan | Correctly translate plan into actuator commands |
| FS_6 | Communicate with other road users | Signal intent so others can predict behavior |
| FS_7 | Detect nominal-performance shortfall | Recognize when not meeting specified performance |
| FD_1 | Ensure operator controllability | Keep a human operator able to take over |
| FD_2 | Detect unavailable degraded mode | Know if the fallback mode has also failed |
| FD_3 | Ensure safe mode transitions | Safe, unambiguous automated/manual transitions |
| FD_4 | React to insufficient performance | Trigger degradation within bounded time once FS_7 fires |
| FD_5 | Reduce performance safely under failure | Define behavior if the degraded mode itself faults |
| FD_6 | Operate within reduced constraints | Execute the tighter degraded envelope until safe stop/handover |

### 8.5 Minimal risk conditions and manoeuvres

**Minimal Risk Condition (MRC):** a tolerable-risk operating state the vehicle can be
brought to. **Minimal Risk Manoeuvre (MRM):** the transition used to reach it — from
operator-takeover request, through comfort stop, to emergency stop. Multiple MRCs/MRMs can
chain depending on remaining capability. Directly applicable to AutoE2E's low-speed urban
maneuvers.

### 8.6 Sense–Plan–Act as the generic architecture

Fail-safe capabilities map onto each Sense–Plan–Act stage; fail-degraded capabilities
layer on top as supervisory/monitoring. Architecturally compatible with AutoE2E's own
reactive-planner / reasoning-band / world-model three-branch structure.

## 9. RSS in Production: Mobileye Drive™ and SuperVision™ {#9-rss-in-production}

Both built on the same stack — REM (crowdsourced HD mapping), RSS as the driving-policy
safety layer, and a redundant multi-modal sensor suite — deployed at different redundancy
and compute levels by automation level and ODD.

**SuperVision™** — "hands-off, eyes-on": lane changes, highway/traffic-jam driving,
point-to-point navigation, evasive manoeuvres; driver must keep eyes on the road. 11
cameras + radar, two EyeQ 5/6 High SoCs. ~300,000 vehicles running it [6].

**Drive™** — "hands-off, eyes-off": driverless robotaxis, ride-pooling, delivery.
Defining difference: **True Redundancy™** — two independent, standalone perception
subsystems (camera-only + separate radar-lidar), each independently capable of driving.
Sample config: four EyeQ 6 High SoCs, eight-plus 8MP cameras, radars, lidars [5].

| Tier | ODD | Compute | Sensing | Autonomy |
|---|---|---|---|---|
| SuperVision | Various (driver engaged) | 2× EyeQ 6 High | 360° camera + radar | Hands-off / eyes-on |
| Drive — Highway | Highway | 3× EyeQ 6 High | + front lidar, surround radar | Hands-off / eyes-off |
| Drive — Arterial/Rural | Arterial & rural | 3× EyeQ 6 High | + imaging radar redundancy, front lidar | Hands-off / eyes-off |
| Drive — Urban | Urban | 4× EyeQ 6 High | + imaging radar redundancy, front lidar | Hands-off / eyes-off |

> The Robotaxi WG's low-speed urban target sits closest to the "urban" tier — where
> Mobileye's own architecture adds the most compute and sensor redundancy relative to
> highway operation.

## 10. Translating This Into Candidate AutoE2E Goals

Proposed starting points for WG discussion, not adopted decisions.

**10.1 An explicit safety envelope around the learned planner.** Define, even informally, a
small set of dangerous-situation triggers (minimum following distance, minimum lateral
clearance on cut-ins, minimum clearance to occluded regions) the output trajectory must
respect, tracked as a metric distinct from raw driving-score benchmarks.

**10.2 Validate against a public scenario taxonomy.** Mirror Mobileye's NHTSA exercise
(§6): pick the NHTSA typology and demonstrate systematically that AutoE2E's behavior (or
its safety envelope) resolves each category the way a human reviewer would expect.

**10.3 Define AutoE2E's own MRC/MRM table.** Given the low-speed urban focus, a small
explicit table: what does AutoE2E do when confidence is low, a sensor branch degrades, or
the reasoning band flags an unresolvable edge case?

**10.4 Redundancy as a longer-horizon architecture question.** True Redundancy is a
Drive-tier concern, beyond AutoE2E's current camera-only scope — but if ambitions extend
toward Drive-tier autonomy claims, redundant, independently-sufficient perception paths
will eventually become a hard requirement.

## 11. AutoE2E Benchmark Plan {#11-autoe2e-benchmark-plan}

Turns §9.4-style V&V thinking into a concrete plan tied to AutoE2E's milestones — the
camera-only, HD-map-free three-band Reactive (10 Hz) / World (1 Hz) / Reasoning (1 Hz)
model, its GRU planning head, and the working group's four milestones: **M1** architecture
(Jul–Aug 2026); **M2** imitation learning (Sep–Oct 2026); **M3** RL policy, closed loop (end
2026); **M4** deployable — ONNX export, quantization, target hardware, demo (Mar–Apr 2027),
split into two reference-design tracks (§12.5).

### 11.1 Open-loop benchmarking — M2 (Sep–Oct 2026)

- **Datasets:** L2D, plus KITScenes for multimodal evaluation.
- **Baselines:** "beat-UniAD" target, VAD and Alpamayo as comparison set.
- **Core metrics:** L2 displacement error (ADE/FDE) at intermediate horizons up to 6.4s;
  collision rate against logged actors; lane-keeping/off-road rate where map-free
  localization permits.
- **Methodological caution:** raw open-loop displacement metrics are inflated by
  "ego-status leakage" — report an ego-status-ablated variant alongside the standard metric.
- **Schedule risk:** M2 compresses IL training, fine-tuning, and open-loop benchmarking into
  a single ~2-month, compute-dependent window. Treat open-loop benchmarking as an internal
  checkpoint with its own short report even though it's not a separate board milestone.
- **Stretch goal:** a NAVSIM/PDMS-style scorecard (no-at-fault collisions, drivable-area
  compliance, TTC, ego progress, comfort) — aspirational, needs log-format reconciliation
  with L2D/KITScenes first.

### 11.2 Closed-loop benchmarking — M3 (end of 2026)

- **Harness:** Bench2Drive/CARLA as primary target (most actively maintained community
  leaderboard, 60+ published methods); NAVSIM v2/HUGSIM as secondary.
- **Core metrics:** CARLA-style Driving Score, Route Completion, Success Rate, infraction
  breakdown (collision, red-light violation, off-road, blocked traffic).
- **Direct tie to §10.1:** CARLA's infraction categories map almost one-to-one onto an
  RSS-style dangerous-situation taxonomy — collision ↔ safe-distance violation, off-road ↔
  lateral safety envelope, red-light/right-of-way ↔ Rule 3. Report infractions against this
  mapping explicitly so closed-loop numbers double as safety evidence.

### 11.3 Driving review — M4 (Mar–Apr 2027)

HIL (target platform, e.g. R-Car) → VIL (reference vehicle) staging currently unconfirmed —
open question (§13). Recommend a standardized per-session log regardless of staging:

- Scenario description and sensor/ODD conditions
- Safety-envelope violation flag (§10.1)
- MRC/MRM triggered, if any (§8.5, §12.2)
- Intervention type and reason (not just a count)
- Which reference-design track (camera-only L2++ or LiDAR-redundant L4) the session used
- Reasoning-band confidence at the moment of intervention, if available — meaningful only
  once the confidence head is actually consumed by the planner (§12.1)

### 11.4 Reporting cadence

With four board-level milestones rather than eight, natural reporting checkpoints no longer
fall out of the structure automatically. Recommend two internal checkpoint reports even
without their own GitHub milestone: open-loop results at end of M2, closed-loop results at
end of M3, each a short table plus one-paragraph interpretation. A third accompanies M4's
demo.

## 12. Where to Solidify the Architecture for Safety

**12.1 Close the reasoning-band confidence loop first.** Issue #103 asked for
horizon/region-aware confidence alongside scenario classification. PR #108 implements
containment correctly (zero-init gate, defaults safe) — but the confidence output is
currently **unsupervised and unconsumed by the planner**: structurally correct, functionally
dead. This is the single highest-leverage safety item: it's exactly the signal the §10.1
envelope needs. Recommend sequencing the follow-up issues (supervise the output;
region/spatial stratification) as a definition of done ahead of or alongside M2.

**12.2 Define AutoE2E's own MRC/MRM table before HIL.** What the system does when the
reasoning band flags low confidence, the world-model rollout diverges from the reactive
branch, or camera input is degraded is currently architecturally undefined. Recommend a
one-page trigger→MRC→MRM table before on-hardware work begins.

**12.3 The camera-only vs. LiDAR-redundant tradeoff is now explicit — good.** The WG's M4
plan targets two reference designs: a low-cost camera-only urban-pilot track (L2++) and a
LiDAR/radar-added, redundant track in C++/ROS (L4 robotaxi, certification as an explicit
goal). §12.5 formalizes what this split means for the safety argument.

**12.4 Priority ordering.** If only one item can be done before M2 closes: close 12.1
first. Already scoped, unblocks the safety envelope, benefits both tracks equally, and
makes every subsequent benchmark number reportable with a confidence dimension attached.

**12.5 Formalizing the two-track safety framing**

| | Track A — Camera-only urban pilot | Track B — LiDAR/radar-redundant |
|---|---|---|
| Target level | L2++ (driver in the loop) | L4 robotaxi |
| Sensing | Camera-only, no redundant modality | Camera + radar + LiDAR, True-Redundancy-style independent paths |
| Governing standard | SOTIF as primary lens; RSS-style envelope strong-to-have | SOTIF + explicit RSS envelope + likely ISO 26262 ASIL decomposition |
| MRC/MRM needs | Lighter — driver takeover is itself a valid MRM | Full table — no driver fallback |
| Confidence-head role | Useful for UX/alerting the driver | Load-bearing — feeds envelope and MRC/MRM triggers directly |
| Certification posture | Documentation-readiness | HARA-equivalent + safety-case draft by end of M4 |

> One architecture, two safety postures — not two separate models and not one blended
> safety claim accurate for neither track.

## 13. References

1. Shalev-Shwartz, S., Shammah, S., Shashua, A. (2017, rev. 2018). "On a Formal Model of
   Safe and Scalable Self-driving Cars." [arXiv:1708.06374](https://arxiv.org/pdf/1708.06374)
2. Mobileye. "Implementing the RSS Model on NHTSA Pre-Crash Scenarios."
   [static.mobileye.com](https://static.mobileye.com/website/corporate/rss/rss_on_nhtsa.pdf)
3. Aptiv, Audi, Baidu, BMW, Continental, Daimler, FCA, HERE, Infineon, Intel, Volkswagen
   (2019). "Safety First for Automated Driving."
   [static.mobileye.com](https://static.mobileye.com/website/corporate/media/Intel-Safety-First-for-Automated-Driving.pdf)
4. Mobileye. "Responsibility-Sensitive Safety (RSS)."
   [mobileye.com](https://www.mobileye.com/technology/responsibility-sensitive-safety/)
5. Mobileye. "Mobileye Drive™." [mobileye.com](https://www.mobileye.com/solutions/drive/)
6. Mobileye. "Mobileye SuperVision™." [mobileye.com](https://www.mobileye.com/solutions/super-vision/)
7. Kalra, N., Paddock, S. M. (2016). "Driving to Safety: How Many Miles of Driving Would It
   Take to Demonstrate Autonomous Vehicle Reliability?" RAND Corporation.

Additional standards referenced but not directly cited as source documents: IEEE 2846 (in
development, sagroups.ieee.org/2846), China ITS Industry Alliance 0116-2019, ISO 4804, ISO
21448 (SOTIF), ISO 26262 (functional safety).