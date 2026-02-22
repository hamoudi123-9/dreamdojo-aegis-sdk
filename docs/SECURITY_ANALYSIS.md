# AEGIS DreamDojo SDK — Security Analysis Report

Based on the NVIDIA DreamDojo Red Team risk assessment conducted by AEGIS.

---

## 1. Target Model: NVIDIA DreamDojo

| Item | Detail |
|------|--------|
| Model Type | Diffusion Transformer (DiT) + Rectified Flow |
| Parameters | 2B / 14B |
| Function | Image + Robot Action → Future video prediction (world simulation) |
| Supported Robots | GR-1, Unitree G1, YAM (bimanual), AgiBot |
| Action Dimensions | 384 (4 robots + MANO + LAM latent) |
| Sub-model | LAM (Latent Action Model) — VAE-based self-supervised action extraction |
| Training Data | 44,000 hours of human egocentric video |

---

## 2. DreamDojo Built-in Guardrails — Limitations

| Guardrail | Method | Limitation |
|-----------|--------|------------|
| Blocklist | Keyword/profanity filter | Easily bypassed (encoding, synonyms) |
| Qwen3Guard (0.6B) | Text safety classifier | Small model, vulnerable to multilingual/context attacks |
| VideoContentSafety | SigLIP + MLP classifier | Only 7 categories; does not inspect robot action safety |
| RetinaFace | Face blurring | Post-processing only, not a defense mechanism |

**Critical design flaw:** `--disable-guardrails` flag disables all safety mechanisms at once. No guardrails exist during the training phase.

---

## 3. Identified Threats and SDK Guard Mapping

### 3.1 CRITICAL — Action Space Manipulation

**Risk:** DreamDojo's 384-dim action vector uses only min-max normalization. If normalization statistics (`shared_meta/*_stats.json`) are exposed, inverse normalization can produce real robot joint commands.

**Attack path:** Manipulated action sequence → DreamDojo inference → "safe-looking" predicted video → dangerous motion executed on physical robot

**SDK Guard:** `ActionSpaceGuard`
- NaN/Inf detection across all dimensions
- Embodiment-specific joint bounds validation (GR-1, G1, YAM, AgiBot)
- Velocity spike detection (delta-from-previous)
- Scale boundary violation (extreme out-of-range values)
- Gripper range validation

```rust
use aegis_dreamdojo_sdk::prelude::*;

let guard = ActionSpaceGuard::new(EmbodimentType::Gr1);
let result = guard.evaluate(&action_tensor);
if !result.is_safe {
    // Block action before sending to robot
}
```

### 3.2 CRITICAL — Prompt Injection via World Model

**Risk:** DreamDojo uses T5/UMT5 text embeddings as conditions. All 7 AEGIS Red Team algorithms (PAIR, Crescendo, AutoDAN, ArtPrompt, HPM, BEAST, TAP) are applicable.

| Attack Algorithm | DreamDojo Scenario | Expected Impact |
|------------------|--------------------|-----------------|
| PAIR | Iterative refinement to bypass Qwen3Guard (0.6B) | Guardrail neutralization |
| Crescendo | Multi-stage escalation to induce dangerous scenarios | Safety threshold evasion |
| AutoDAN | Genetic algorithm for adversarial text suffix | Automated mass attacks |
| ArtPrompt | ASCII art encoding to bypass blocklist | Text filter neutralization |
| HPM | Psychological framing to justify dangerous simulation | Up to 88.1% ASR |
| BEAST | Beam search for adversarial token suffixes | Token-level attack |
| TAP | Tree-of-attacks for efficient attack path search | Maximized search efficiency |

**SDK Guard:** `GuidanceGuard` (parameter validation layer)
- Not a text-level defense, but validates inference parameters that amplify prompt injection effects
- Extreme guidance scale detection prevents amplification of adversarial prompts
- **Full prompt-level defense requires AEGIS Pro PALADIN 6-Layer system**

### 3.3 HIGH — Multimodal Visual Attack

**Risk:** DreamDojo's condition frames (robot camera inputs) are vulnerable to physical-world visual attacks.

| Attack | Vector | ASR |
|--------|--------|-----|
| VisCRA | Attention masking image in robot's field of view | 76.48% |
| MML | Cross-modal encrypted commands in condition frame | 99.40% |
| OCR Injection | Text placed in environment read by model | — |
| Adversarial Perturbation | Imperceptible noise distorting condition frame | — |
| Steganography | LSB steganography with hidden commands | — |

**SDK Guard:** `WorldModelInputGuard`
- All-zero frame detection (blank/corrupted input)
- Uniform color detection (solid color injection)
- Mean/std anomaly detection (too dark, too bright, extreme variance)
- Basic high-frequency noise detection (adjacent pixel energy)
- Dimension validation

```rust
let guard = WorldModelInputGuard::new();
let result = guard.evaluate(&condition_frame);
// Catches blank frames, uniform injection, basic perturbation
```

> **Note:** SDK provides basic statistical checks. AEGIS Pro adds 2nd-order finite difference spectral analysis and gradient magnitude analysis for stronger adversarial detection.

### 3.4 HIGH — Autoregressive Chain Attack (Sliding Window Poisoning)

**Risk:** DreamDojo uses autoregressive generation where the last generated frame becomes the next condition frame:

```
Frame_0 → [generate] → Frame_1~12 → Frame_12 as condition → [generate] → Frame_13~24 → ...
```

Initial subtle perturbation accumulates through the chain (error accumulation), producing predictions completely diverged from reality in long simulations.

**SDK Guard:** `AutoregressiveChainGuard`
- Empty chain detection
- Chain length limit enforcement (default: 64 frames)
- Temporal delta spike detection (sudden jumps between frames)
- Cumulative drift tracking (sum of deltas exceeding threshold)

```rust
let guard = AutoregressiveChainGuard::new();
let result = guard.evaluate(&predicted_frames);
// Detects error accumulation and temporal discontinuities
```

> **Note:** SDK uses simple cumulative sum for drift. AEGIS Pro uses Welford online algorithm for streaming variance-based drift detection with statistical anomaly (z-score) analysis.

### 3.5 HIGH — LAM (Latent Action Model) Manipulation

**Risk:** LAM extracts implicit actions in a 32-dim latent space with extremely small KL weight (β = 0.000001), resulting in a nearly deterministic VAE with poorly regularized latent space. DreamDojo's dimensions [352:384] are allocated to LAM latent — direct manipulation can produce physically impossible motions.

**SDK Guard:** `LatentSpaceGuard`
- NaN/Inf detection in latent vectors
- Dimension validation (expected: 32)
- L2 norm bounds (collapsed/exploding representation detection)
- Element-wise sigma analysis (individual dimension outliers)

```rust
let guard = LatentSpaceGuard::new();
let result = guard.evaluate(&latent_vector);
// Catches collapsed, exploding, or anomalous latent representations
```

> **Note:** SDK provides basic norm and sigma checks. AEGIS Pro adds Pearson sliding-window correlation analysis to detect adversarial latent patterns across batch samples.

### 3.6 HIGH — Training Data Poisoning

**Risk:**
- 44K hours of internet video — no integrity verification
- LeRobot/HuggingFace datasets can be poisoned before fine-tuning
- `webdataset_s3.py` loads directly from S3 paths — supply chain attack vector

**SDK Coverage:** Out of scope (runtime validation only). AEGIS Pro provides Supply Chain Security with SHA-256 integrity verification and typosquatting detection.

### 3.7 MEDIUM — Classifier-Free Guidance Abuse

**Risk:** `velocity_fn = cond_v + guidance * (cond_v - uncond_v)` — extreme guidance values produce amplified predictions that appear unrealistic but are executable by robots.

**SDK Guard:** `GuidanceGuard`
- Guidance scale range validation (default: 1.0–15.0)
- Extreme guidance detection (threshold: 50.0 in SDK, 20.0 in Pro)
- Step count validation
- Resolution and conditional frame count validation

### 3.8 MEDIUM — Autonomy Boundary Violation

**Risk:** When DreamDojo serves as a planning module for autonomous robots:
- World model predictions trusted without human-in-the-loop
- Autonomous decisions during communication blackout
- 14B model inference latency causing watchdog timeout
- Military applications lacking rules-of-engagement compliance

**SDK Coverage:** Not directly addressed. AEGIS Pro provides Autonomy Guard, Dead Man Switch, and ROE Compliance modules.

---

## 4. Threat-to-Guard Coverage Matrix

| # | Threat | Severity | SDK Guard | SDK Coverage | Pro Coverage |
|---|--------|----------|-----------|-------------|-------------|
| 1 | Action Space Manipulation | CRITICAL | `ActionSpaceGuard` | Basic bounds + NaN/Inf | CBF Engine + Action Validator |
| 2 | Prompt Injection | CRITICAL | `GuidanceGuard` (partial) | Parameter validation only | PALADIN 6-Layer + 7 Red Team |
| 3 | Visual Adversarial Attack | HIGH | `WorldModelInputGuard` | Statistical checks | Spectral analysis + VisCRA + MML |
| 4 | Autoregressive Error Accumulation | HIGH | `AutoregressiveChainGuard` | Cumulative drift | Welford variance + z-score |
| 5 | LAM Latent Manipulation | HIGH | `LatentSpaceGuard` | Norm + sigma checks | Pearson correlation analysis |
| 6 | Training Data Poisoning | HIGH | — | Not covered | Supply Chain Security |
| 7 | CFG Parameter Abuse | MEDIUM | `GuidanceGuard` | Range validation | Deterministic Defense (θ=0) |
| 8 | Autonomy Boundary Violation | MEDIUM | — | Not covered | Autonomy Guard + HITL |
| 9 | Guardrail Disable Flag | HIGH | — | Not covered | Architectural enforcement |

**SDK covers 6 of 9 threats** at basic validation level.
**AEGIS Pro covers 7 of 9 threats** with production-grade detection.

---

## 5. Recommended Defense Architecture

```
Robot Camera Input
│
▼
┌─────────────────────────────────────┐
│ WorldModelInputGuard (SDK)          │ ← Input frame anomaly detection
│ + AEGIS VCS-M (Pro)                 │   (adversarial pixel, blank frame)
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│ GuidanceGuard (SDK)                 │ ← Inference parameter validation
│ + AEGIS PALADIN (Pro)               │   (prompt injection, CFG abuse)
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│ LatentSpaceGuard (SDK)              │ ← LAM latent vector validation
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│ DreamDojo Inference                 │ ← World model video generation
│ (2B / 14B DiT)                      │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│ AutoregressiveChainGuard (SDK)      │ ← Chain drift & temporal anomaly
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│ ActionSpaceGuard (SDK)              │ ← Generated action safety validation
│ + AEGIS VLA CBF Engine (Pro)        │   (bounds, velocity, physics)
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│ AEGIS Autonomy Guard (Pro)          │ ← Autonomy level gating + HITL
│ + Dead Man Switch                   │
└──────────────┬──────────────────────┘
               ▼
Robot Execution
```

---

## 6. SDK Integration Example

```rust
use aegis_dreamdojo_sdk::prelude::*;

fn validate_dreamdojo_cycle(
    action: &ActionTensor,
    frame: &ConditionFrame,
    params: &InferenceParams,
    latent: &LatentVector,
    chain: &[PredictedFrame],
) -> PipelineResult {
    let pipeline = DreamDojoPipeline::for_gr1();

    let request = PipelineRequest {
        action: Some(action.clone()),
        condition_frame: Some(frame.clone()),
        inference_params: Some(params.clone()),
        latent_vector: Some(latent.clone()),
        predicted_chain: Some(chain.to_vec()),
    };

    let result = pipeline.evaluate(&request);

    match result.risk_level {
        RiskLevel::Safe => println!("SAFE — proceed with execution"),
        RiskLevel::Warning => println!("WARNING — review guard reports"),
        RiskLevel::Danger => println!("DANGER — human review required"),
        RiskLevel::Blocked => println!("BLOCKED — abort execution"),
    }

    for report in &result.guard_reports {
        if !report.passed {
            eprintln!("[FAIL] {}: {}", report.guard_name, report.messages.join("; "));
        }
    }

    result
}
```

---

## 7. Conclusion

DreamDojo excels at world simulation but has critical safety gaps:

1. **Text-only guardrails** — no action/physics safety validation
2. **Guardrail disable flag** — `--disable-guardrails` removes all protection
3. **No input validation** — only output filtering, defenseless against adversarial inputs
4. **No training pipeline security** — 44K hours of internet data unverified

The AEGIS DreamDojo SDK provides a **first line of defense** with 5 guards covering 6 of 9 identified threats at a basic validation level. For production robotics deployments, upgrade to **AEGIS Pro** for optimized thresholds, advanced statistical analysis, and full defense coverage.

| Deployment | Recommended |
|------------|------------|
| Research / Prototyping | SDK (this package) |
| Simulation-only | SDK + custom thresholds |
| Physical robot (lab) | AEGIS Pro required |
| Physical robot (production) | AEGIS Pro + VLA CBF + Autonomy Guard |
| Military / safety-critical | AEGIS Pro + Military Defense modules |
