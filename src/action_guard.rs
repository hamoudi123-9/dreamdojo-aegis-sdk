use std::time::Instant;

use crate::traits::GuardEngine;
use crate::types::*;

/// ActionSpaceGuard validates robot action tensors against embodiment-specific
/// safety bounds. SDK version provides basic NaN/Inf, dimension, range, and
/// zero-action checks.
pub struct ActionSpaceGuard {
    profile: EmbodimentProfile,
    config: ActionGuardConfig,
}

impl ActionSpaceGuard {
    pub fn new(embodiment_type: EmbodimentType) -> Self {
        Self {
            profile: EmbodimentProfile::for_type(embodiment_type),
            config: ActionGuardConfig::default(),
        }
    }

    pub fn with_config(embodiment_type: EmbodimentType, config: ActionGuardConfig) -> Self {
        Self {
            profile: EmbodimentProfile::for_type(embodiment_type),
            config,
        }
    }

    pub fn with_profile(profile: EmbodimentProfile, config: ActionGuardConfig) -> Self {
        Self { profile, config }
    }

    pub fn profile(&self) -> &EmbodimentProfile {
        &self.profile
    }

    fn check_nan_inf(&self, action: &ActionTensor) -> Vec<ActionViolation> {
        let mut violations = Vec::new();
        for (i, v) in action.values.iter().enumerate() {
            if v.is_nan() || v.is_infinite() {
                violations.push(ActionViolation {
                    category: ActionViolationCategory::NanInf,
                    dimension: Some(i),
                    message: format!("Dimension {} contains NaN/Inf: {}", i, v),
                    severity: 1.0,
                });
            }
        }
        violations
    }

    fn check_dimension(&self, action: &ActionTensor) -> Vec<ActionViolation> {
        if action.values.len() != self.profile.action_dim {
            vec![ActionViolation {
                category: ActionViolationCategory::DimensionMismatch,
                dimension: None,
                message: format!(
                    "Expected {} dimensions for {}, got {}",
                    self.profile.action_dim, self.profile.name, action.values.len()
                ),
                severity: 1.0,
            }]
        } else {
            vec![]
        }
    }

    fn check_range(&self, action: &ActionTensor) -> Vec<ActionViolation> {
        let mut violations = Vec::new();
        let multiplier = self.config.scale_boundary_multiplier;

        for (i, v) in action.values.iter().enumerate() {
            if i >= self.profile.dim_bounds.len() {
                break;
            }
            let bound = &self.profile.dim_bounds[i];
            let range = (bound.upper - bound.lower).abs();
            let scaled_lower = bound.lower - range * multiplier;
            let scaled_upper = bound.upper + range * multiplier;

            if *v < scaled_lower || *v > scaled_upper {
                violations.push(ActionViolation {
                    category: ActionViolationCategory::ScaleBoundary,
                    dimension: Some(i),
                    message: format!(
                        "Dimension {} ({}) value {:.4} exceeds scaled bounds [{:.2}, {:.2}]",
                        i, bound.name, v, scaled_lower, scaled_upper
                    ),
                    severity: 0.9,
                });
            } else if *v < bound.lower || *v > bound.upper {
                violations.push(ActionViolation {
                    category: ActionViolationCategory::RangeViolation,
                    dimension: Some(i),
                    message: format!(
                        "Dimension {} ({}) value {:.4} outside bounds [{:.2}, {:.2}]",
                        i, bound.name, v, bound.lower, bound.upper
                    ),
                    severity: 0.5,
                });
            }
        }
        violations
    }

    fn check_zero_action(&self, action: &ActionTensor) -> Vec<ActionViolation> {
        if self.config.allow_zero_action {
            return vec![];
        }
        if action.values.iter().all(|v| *v == 0.0) {
            vec![ActionViolation {
                category: ActionViolationCategory::ZeroAction,
                dimension: None,
                message: "All-zero action detected".into(),
                severity: 0.3,
            }]
        } else {
            vec![]
        }
    }

    fn check_velocity(&self, action: &ActionTensor) -> Vec<ActionViolation> {
        let prev = match &action.previous_values {
            Some(p) if p.len() == action.values.len() => p,
            _ => return vec![],
        };

        let mut violations = Vec::new();
        for (i, (cur, prv)) in action.values.iter().zip(prev.iter()).enumerate() {
            if i >= self.profile.max_velocity.len() {
                break;
            }
            let delta = (cur - prv).abs();
            if delta > self.profile.max_velocity[i] {
                violations.push(ActionViolation {
                    category: ActionViolationCategory::VelocitySpike,
                    dimension: Some(i),
                    message: format!(
                        "Dimension {} velocity {:.4} exceeds max {:.2}",
                        i, delta, self.profile.max_velocity[i]
                    ),
                    severity: 0.7,
                });
            }
        }
        violations
    }

    fn check_gripper(&self, action: &ActionTensor) -> Vec<ActionViolation> {
        let mut violations = Vec::new();
        let (gmin, gmax) = self.profile.gripper_range;

        for &idx in &self.profile.gripper_indices {
            if idx >= action.values.len() {
                continue;
            }
            let v = action.values[idx];
            if v < gmin || v > gmax {
                violations.push(ActionViolation {
                    category: ActionViolationCategory::GripperRange,
                    dimension: Some(idx),
                    message: format!(
                        "Gripper dimension {} value {:.4} outside range [{:.2}, {:.2}]",
                        idx, v, gmin, gmax
                    ),
                    severity: 0.6,
                });
            }
        }
        violations
    }
}

impl GuardEngine for ActionSpaceGuard {
    type Input = ActionTensor;
    type Output = ActionGuardResult;

    fn evaluate(&self, action: &ActionTensor) -> ActionGuardResult {
        let start = Instant::now();
        let mut violations = Vec::new();

        // NaN/Inf first — critical
        violations.extend(self.check_nan_inf(action));
        // Dimension check
        let dim_violations = self.check_dimension(action);
        let dim_mismatch = !dim_violations.is_empty();
        violations.extend(dim_violations);

        // Only run value checks if dimensions match
        if !dim_mismatch {
            violations.extend(self.check_range(action));
            violations.extend(self.check_zero_action(action));
            violations.extend(self.check_velocity(action));
            violations.extend(self.check_gripper(action));
        }

        let messages: Vec<String> = violations.iter().map(|v| v.message.clone()).collect();

        // SDK risk scoring: max(severities) — simplified from Pro
        let risk_score = violations
            .iter()
            .map(|v| v.severity)
            .fold(0.0_f64, f64::max);

        let is_safe = risk_score < self.config.block_threshold;

        ActionGuardResult {
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
    fn test_valid_gr1_action() {
        let guard = ActionSpaceGuard::new(EmbodimentType::Gr1);
        let action = ActionTensor {
            values: vec![0.1, -0.2, 0.0, 0.5, -0.3, 0.8, 0.2],
            previous_values: None,
            embodiment_type: EmbodimentType::Gr1,
            step_index: 0,
            recent_gripper_states: vec![],
        };
        let result = guard.evaluate(&action);
        assert!(result.is_safe);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn test_nan_detection() {
        let guard = ActionSpaceGuard::new(EmbodimentType::Gr1);
        let action = ActionTensor {
            values: vec![0.1, f64::NAN, 0.0, 0.5, -0.3, 0.8, 0.2],
            previous_values: None,
            embodiment_type: EmbodimentType::Gr1,
            step_index: 0,
            recent_gripper_states: vec![],
        };
        let result = guard.evaluate(&action);
        assert!(!result.is_safe);
        assert_eq!(result.violations[0].category, ActionViolationCategory::NanInf);
    }

    #[test]
    fn test_dimension_mismatch() {
        let guard = ActionSpaceGuard::new(EmbodimentType::Gr1);
        let action = ActionTensor {
            values: vec![0.1, 0.2],
            previous_values: None,
            embodiment_type: EmbodimentType::Gr1,
            step_index: 0,
            recent_gripper_states: vec![],
        };
        let result = guard.evaluate(&action);
        assert!(!result.is_safe);
        assert_eq!(
            result.violations[0].category,
            ActionViolationCategory::DimensionMismatch
        );
    }

    #[test]
    fn test_zero_action() {
        let guard = ActionSpaceGuard::new(EmbodimentType::Gr1);
        let action = ActionTensor {
            values: vec![0.0; 7],
            previous_values: None,
            embodiment_type: EmbodimentType::Gr1,
            step_index: 0,
            recent_gripper_states: vec![],
        };
        let result = guard.evaluate(&action);
        assert!(result.is_safe); // severity 0.3 < 0.8
        assert_eq!(
            result.violations[0].category,
            ActionViolationCategory::ZeroAction
        );
    }
}
