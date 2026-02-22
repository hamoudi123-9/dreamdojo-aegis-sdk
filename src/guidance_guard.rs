use std::time::Instant;

use crate::traits::GuardEngine;
use crate::types::*;

/// GuidanceGuard validates inference parameters for world model generation.
/// SDK version provides basic range validation for guidance scale, step count,
/// conditional frames, and resolution. Pro version includes suspicious seed
/// detection and mode collapse analysis.
pub struct GuidanceGuard {
    config: GuidanceGuardConfig,
}

impl GuidanceGuard {
    pub fn new() -> Self {
        Self {
            config: GuidanceGuardConfig::default(),
        }
    }

    pub fn with_config(config: GuidanceGuardConfig) -> Self {
        Self { config }
    }

    fn check_guidance_scale(&self, params: &InferenceParams) -> Vec<GuidanceViolation> {
        let mut violations = Vec::new();
        let g = params.guidance_scale;

        if g < self.config.min_guidance || g > self.config.max_guidance {
            violations.push(GuidanceViolation {
                category: GuidanceViolationCategory::GuidanceScale,
                message: format!(
                    "Guidance scale {:.2} outside valid range [{:.1}, {:.1}]",
                    g, self.config.min_guidance, self.config.max_guidance
                ),
                severity: 0.6,
            });
        }

        if g > self.config.extreme_guidance {
            violations.push(GuidanceViolation {
                category: GuidanceViolationCategory::ExtremeGuidance,
                message: format!(
                    "Extreme guidance scale {:.2} (threshold: {:.1}) — risk of mode collapse",
                    g, self.config.extreme_guidance
                ),
                severity: 0.9,
            });
        }

        violations
    }

    fn check_steps(&self, params: &InferenceParams) -> Vec<GuidanceViolation> {
        if params.num_steps < self.config.min_steps || params.num_steps > self.config.max_steps {
            vec![GuidanceViolation {
                category: GuidanceViolationCategory::StepCount,
                message: format!(
                    "Step count {} outside valid range [{}, {}]",
                    params.num_steps, self.config.min_steps, self.config.max_steps
                ),
                severity: 0.5,
            }]
        } else {
            vec![]
        }
    }

    fn check_conditional_frames(&self, params: &InferenceParams) -> Vec<GuidanceViolation> {
        if !self.config.allowed_conditional_frames.contains(&params.num_conditional_frames) {
            vec![GuidanceViolation {
                category: GuidanceViolationCategory::ConditionalFrames,
                message: format!(
                    "Conditional frame count {} not in allowed set {:?}",
                    params.num_conditional_frames, self.config.allowed_conditional_frames
                ),
                severity: 0.4,
            }]
        } else {
            vec![]
        }
    }

    fn check_resolution(&self, params: &InferenceParams) -> Vec<GuidanceViolation> {
        if !self.config.allowed_resolutions.contains(&params.resolution) {
            vec![GuidanceViolation {
                category: GuidanceViolationCategory::Resolution,
                message: format!(
                    "Resolution {:?} not in allowed set {:?}",
                    params.resolution, self.config.allowed_resolutions
                ),
                severity: 0.4,
            }]
        } else {
            vec![]
        }
    }
}

impl Default for GuidanceGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl GuardEngine for GuidanceGuard {
    type Input = InferenceParams;
    type Output = GuidanceGuardResult;

    fn evaluate(&self, params: &InferenceParams) -> GuidanceGuardResult {
        let start = Instant::now();
        let mut violations = Vec::new();

        violations.extend(self.check_guidance_scale(params));
        violations.extend(self.check_steps(params));
        violations.extend(self.check_conditional_frames(params));
        violations.extend(self.check_resolution(params));

        let messages: Vec<String> = violations.iter().map(|v| v.message.clone()).collect();
        let risk_score = violations
            .iter()
            .map(|v| v.severity)
            .fold(0.0_f64, f64::max);
        let is_safe = risk_score < self.config.block_threshold;

        GuidanceGuardResult {
            is_safe,
            risk_score,
            violations,
            messages,
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_params() {
        let guard = GuidanceGuard::new();
        let params = InferenceParams {
            guidance_scale: 7.5,
            num_steps: 50,
            num_conditional_frames: 2,
            resolution: [512, 512],
            seed: None,
            use_negative_prompt: false,
        };
        let result = guard.evaluate(&params);
        assert!(result.is_safe);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn test_extreme_guidance() {
        let guard = GuidanceGuard::new();
        let params = InferenceParams {
            guidance_scale: 100.0,
            num_steps: 50,
            num_conditional_frames: 2,
            resolution: [512, 512],
            seed: None,
            use_negative_prompt: false,
        };
        let result = guard.evaluate(&params);
        assert!(!result.is_safe);
    }

    #[test]
    fn test_invalid_resolution() {
        let guard = GuidanceGuard::new();
        let params = InferenceParams {
            guidance_scale: 7.5,
            num_steps: 50,
            num_conditional_frames: 2,
            resolution: [999, 999],
            seed: None,
            use_negative_prompt: false,
        };
        let result = guard.evaluate(&params);
        assert!(result.violations.iter().any(|v| v.category == GuidanceViolationCategory::Resolution));
    }
}
