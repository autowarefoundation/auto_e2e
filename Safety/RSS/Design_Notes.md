# Safety Envelope — Design Notes


## RSS rule → formal mechanism (reference)

| Rule | Informal statement | Formal mechanism |
|---|---|---|
| 1. Safe distance | Don't hit the car in front of you. | Minimum longitudinal gap from both cars' velocities and worst-case braking/acceleration. |
| 2. No reckless cut-ins | Don't cut into another car's safety envelope. | Minimum lateral gap, respected on lane changes/merges. |
| 3. Right-of-way given, not taken | Priority doesn't excuse causing a collision. | The prioritized agent must still brake if the other can't safely stop. |
| 4. Caution under limited visibility | Be careful where you can't see. | "Exposure time" — respond safely from the first moment an object could be seen. |
| 5. Avoid a crash if you can without causing another | If you can prevent an accident safely, you must. | A legal evasive manoeuvre is required whenever one exists that doesn't itself violate rules 1–4. |

## Envelope triggers to implement

- [ ] Minimum following distance (longitudinal safe-distance trigger, RSS Rule 1)
- [ ] Minimum lateral clearance on cut-ins (RSS Rule 2)
- [ ] Minimum clearance to occluded regions (RSS Rule 4 / exposure time)
- [ ] Violation logging as a distinct metric, separate from driving-score benchmarks
- [ ] Per-confidence-bucket reporting once the confidence head (above) is live

Engineering-level approximations of rules 1, 2, and 4, chosen because they're the three most directly checkable against a predicted trajectory without
needing a full multi-agent formal model.

## Track-dependent strictness

| | Track A — Camera-only urban pilot (L2++) | Track B — LiDAR/radar-redundant (L4) |
|---|---|---|
| Envelope role | Strong-to-have; driver remains the fallback | Load-bearing; no human fallback |
| Governing standard | SOTIF primary | SOTIF + explicit RSS envelope + likely ISO 26262 ASIL |