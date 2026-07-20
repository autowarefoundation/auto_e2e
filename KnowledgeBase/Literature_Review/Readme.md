# AutoE2E -- Literature Review of E2E models

[Literature Review of Vision Language Action Models for Autonomous Driving (End to End Model for Autonomous Driving)](https://docs.google.com/document/d/1ls-7LhYuCL1SbgiJBoCRg62UZ9FdcgLv5uA2zSYTIkY/edit?usp=sharing)

## Proposed structure:

1. Introduction (Move from field-level evolution → technological inflection → research need → contribution)
  
  a) Evolution of Autonomous Driving Architectures (establish the paradigm shift)
    - Brief history of modular pipelines (perception → prediction → planning → control)
    - Strengths: interpretability, component validation
    - Weaknesses: error propagation, brittle interfaces, scaling limits
  
  b) Emergence of Vision-Language Models and its applications (why language suddenly matters in driving)
    - Cone effect: AI, DL, FM, MM, VLM, VLA
    - semantic grounding, instruction following, reasoning scaffolds, supervision via text
    - Applications GenAI : world modeling, scenario generation, synthetic data engines, trajectory diffusion, simulation realism
    - Applications for real implementation of 
  
  c) Motivation for End-to-End Driving (What problems does it attempt to solve?) - (E2E for VisionPilot PRO: a single monolithic neural network can map sensing data to safe driving trajectories — essentially learning the entire driving task).
    - error propagation in modular stacks
    - hand-engineered interfaces
    - limited generalization
  
  d) Objective and Scope of This Review (examines the lifecycle of systems — design through deployment)

2. Evaluation Framework 
  
  a) Scope of the Review (defining boundaries what to include and exclude, to know we should ask: Does the model participate directly in the driving decision loop?)
    - Include:
      - end-to-end driving architectures
      - VLA / multimodal foundation models
      - learned planners
    - Exclude:
      - purely modular stacks
      - perception-only models
      - rule-based planners
        (Mental Mode : 
        Level 1 — Sensor Encoding (CNN / ViT / ConvNeXt / etc.) => Supporting infrastructure, not in scope)
        Level 2 — Scene Representation (BEV, tokens, latent world state.)
        Level 3 — Decision Generation (trajectory, control, action tokens.)
  
  b) Evaluation Dimensions (Lifecycle)
      - Design: Architecture philosophy, sensor fusion, representation strategy.
      - Training: Datasets, supervision signals, scaling behavior.
      - Operationalization: Real-world deployment maturity, closed-loop capability, inference constraints.
      - Validation: Metrics, safety evidence, robustness testing.
      - Scalability: Training compute, data engine requirements, economic feasibility.
  
  c) Taxonomy of End-to-End Architectures
    - Reactive Vision-to-Control
      * Direct regression from pixels to control.
      * Low latency, limited interpretability.
    - Transformer-Based World Models
      * Scene tokenization and temporal reasoning.
      * Better context modeling, higher compute.
    - Vision-Language-Action Planners
    Token-based decision generation.
      * Flexible supervision, emerging safety questions.
    - Reasoning-Augmented Models
    - Generative Trajectory Models
    - Mixture-of-Experts Architectures
  
  d) Building and Training a VLM (smoke test, forward passes)

3. Systematic Review of Existing Solutions (organized by taxonomy category)
 - Reactive Vision-to-Control
 - Transformer-Based World Models
 - Vision-Language-Action Planners
 - Reasoning-Augmented Models
 - Generative Trajectory Models
 - Mixture-of-Experts Architectures

=> For EACH model, define : 
 - System Overview: Architecture + inputs + outputs.
 - Training Strategy: Datasets, supervision, scaling.
 - Deployment Status: Simulation, closed-loop, real-world.
 - Validation and Safety Evidence
 - Compute Profile: Training scale if known. Inference feasibility if reported.
 - Key Insight (comment on the contribution of the models)

4. Cross-System Synthesis and Research Trajectories
 - Architectural Convergence
 - Validation Bottlenecks (Large monolithic networks are “difficult to validate” because internal representations are not explicit, Imitation learning may replicate unsafe human behavior), Norms
 - Compute Trajectories
 - Implications for Next-Generation Systems
