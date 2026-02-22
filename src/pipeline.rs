use std::time::Instant;

use crate::action_guard::ActionSpaceGuard;
use crate::chain_guard::AutoregressiveChainGuard;
use crate::guidance_guard::GuidanceGuard;
use crate::input_guard::WorldModelInputGuard;
use crate::latent_guard::LatentSpaceGuard;
use crate::traits::GuardEngine;
use crate::types::*;

/// DreamDojoPipeline orchestrates all five guards in sequence.
///
/// Execution order:
/// 1. **Pre-inference:** Action → Input → Guidance → Latent
/// 2. **Post-inference:** Chain
///
/// SDK risk aggregation: `max(scores)` (Pro uses weighted `max*0.6 + mean*0.4`)
pub struct DreamDojoPipeline {
    action_guard: ActionSpaceGuard,
    input_guard: WorldModelInputGuard,
    guidance_guard: GuidanceGuard,
    latent_guard: LatentSpaceGuard,
    chain_guard: AutoregressiveChainGuard,
    early_exit: bool,
}

impl DreamDojoPipeline {
    pub fn new(embodiment_type: EmbodimentType) -> Self {
        Self {
            action_guard: ActionSpaceGuard::new(embodiment_type),
            input_guard: WorldModelInputGuard::new(),
            guidance_guard: GuidanceGuard::new(),
            latent_guard: LatentSpaceGuard::new(),
            chain_guard: AutoregressiveChainGuard::new(),
            early_exit: false,
        }
    }

    pub fn with_config(embodiment_type: EmbodimentType, config: PipelineConfig) -> Self {
        Self {
            action_guard: ActionSpaceGuard::with_config(embodiment_type, config.action_config),
            input_guard: WorldModelInputGuard::with_config(config.input_config),
            guidance_guard: GuidanceGuard::with_config(config.guidance_config),
            latent_guard: LatentSpaceGuard::with_config(config.latent_config),
            chain_guard: AutoregressiveChainGuard::with_config(config.chain_config),
            early_exit: config.early_exit,
        }
    }

    pub fn for_gr1() -> Self {
        Self::new(EmbodimentType::Gr1)
    }

    pub fn for_g1() -> Self {
        Self::new(EmbodimentType::G1)
    }

    pub fn evaluate(&self, request: &PipelineRequest) -> PipelineResult {
        let start = Instant::now();
        let mut guard_reports = Vec::new();
        let mut action_result = None;
        let mut input_result = None;
        let mut guidance_result = None;
        let mut latent_result = None;
        let mut chain_result = None;

        // 1. Action Guard
        if let Some(ref action) = request.action {
            let result = self.action_guard.evaluate(action);
            guard_reports.push(GuardReport {
                guard_name: "ActionSpaceGuard".into(),
                risk_score: result.risk_score,
                passed: result.is_safe,
                violation_count: result.violations.len(),
                messages: result.messages.clone(),
            });
            let blocked = !result.is_safe;
            action_result = Some(result);
            if self.early_exit && blocked {
                return self.build_result(
                    guard_reports, action_result, input_result,
                    guidance_result, latent_result, chain_result, start,
                );
            }
        }

        // 2. Input Guard
        if let Some(ref frame) = request.condition_frame {
            let result = self.input_guard.evaluate(frame);
            guard_reports.push(GuardReport {
                guard_name: "WorldModelInputGuard".into(),
                risk_score: result.risk_score,
                passed: result.is_safe,
                violation_count: result.violations.len(),
                messages: result.messages.clone(),
            });
            let blocked = !result.is_safe;
            input_result = Some(result);
            if self.early_exit && blocked {
                return self.build_result(
                    guard_reports, action_result, input_result,
                    guidance_result, latent_result, chain_result, start,
                );
            }
        }

        // 3. Guidance Guard
        if let Some(ref params) = request.inference_params {
            let result = self.guidance_guard.evaluate(params);
            guard_reports.push(GuardReport {
                guard_name: "GuidanceGuard".into(),
                risk_score: result.risk_score,
                passed: result.is_safe,
                violation_count: result.violations.len(),
                messages: result.messages.clone(),
            });
            let blocked = !result.is_safe;
            guidance_result = Some(result);
            if self.early_exit && blocked {
                return self.build_result(
                    guard_reports, action_result, input_result,
                    guidance_result, latent_result, chain_result, start,
                );
            }
        }

        // 4. Latent Guard
        if let Some(ref latent) = request.latent_vector {
            let result = self.latent_guard.evaluate(latent);
            guard_reports.push(GuardReport {
                guard_name: "LatentSpaceGuard".into(),
                risk_score: result.risk_score,
                passed: result.is_safe,
                violation_count: result.violations.len(),
                messages: result.messages.clone(),
            });
            let blocked = !result.is_safe;
            latent_result = Some(result);
            if self.early_exit && blocked {
                return self.build_result(
                    guard_reports, action_result, input_result,
                    guidance_result, latent_result, chain_result, start,
                );
            }
        }

        // 5. Chain Guard (post-inference)
        if let Some(ref chain) = request.predicted_chain {
            let result = self.chain_guard.evaluate(chain);
            guard_reports.push(GuardReport {
                guard_name: "AutoregressiveChainGuard".into(),
                risk_score: result.risk_score,
                passed: result.is_safe,
                violation_count: result.violations.len(),
                messages: result.messages.clone(),
            });
            chain_result = Some(result);
        }

        self.build_result(
            guard_reports, action_result, input_result,
            guidance_result, latent_result, chain_result, start,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build_result(
        &self,
        guard_reports: Vec<GuardReport>,
        action_result: Option<ActionGuardResult>,
        input_result: Option<InputGuardResult>,
        guidance_result: Option<GuidanceGuardResult>,
        latent_result: Option<LatentGuardResult>,
        chain_result: Option<ChainGuardResult>,
        start: Instant,
    ) -> PipelineResult {
        let scores: Vec<f64> = guard_reports.iter().map(|r| r.risk_score).collect();

        // SDK aggregation: simple max (Pro uses max*0.6 + mean*0.4)
        let overall_risk = scores.iter().copied().fold(0.0_f64, f64::max);

        let risk_level = RiskLevel::from_score(overall_risk);
        let is_safe = matches!(risk_level, RiskLevel::Safe | RiskLevel::Warning);

        PipelineResult {
            is_safe,
            overall_risk,
            risk_level,
            guard_reports,
            action_result,
            input_result,
            guidance_result,
            latent_result,
            chain_result,
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_action_only() {
        let pipeline = DreamDojoPipeline::for_gr1();
        let request = PipelineRequest {
            action: Some(ActionTensor {
                values: vec![0.1, -0.2, 0.0, 0.5, -0.3, 0.8, 0.2],
                previous_values: None,
                embodiment_type: EmbodimentType::Gr1,
                step_index: 0,
                recent_gripper_states: vec![],
            }),
            condition_frame: None,
            inference_params: None,
            latent_vector: None,
            predicted_chain: None,
        };
        let result = pipeline.evaluate(&request);
        assert!(result.is_safe);
        assert_eq!(result.guard_reports.len(), 1);
    }

    #[test]
    fn test_pipeline_empty_request() {
        let pipeline = DreamDojoPipeline::for_gr1();
        let request = PipelineRequest {
            action: None,
            condition_frame: None,
            inference_params: None,
            latent_vector: None,
            predicted_chain: None,
        };
        let result = pipeline.evaluate(&request);
        assert!(result.is_safe);
        assert!(result.guard_reports.is_empty());
    }

    #[test]
    fn test_pipeline_nan_action_blocks() {
        let pipeline = DreamDojoPipeline::for_gr1();
        let request = PipelineRequest {
            action: Some(ActionTensor {
                values: vec![f64::NAN; 7],
                previous_values: None,
                embodiment_type: EmbodimentType::Gr1,
                step_index: 0,
                recent_gripper_states: vec![],
            }),
            condition_frame: None,
            inference_params: None,
            latent_vector: None,
            predicted_chain: None,
        };
        let result = pipeline.evaluate(&request);
        assert!(!result.is_safe);
        assert_eq!(result.risk_level, RiskLevel::Blocked);
    }

    #[test]
    fn test_pipeline_full_safe() {
        let pipeline = DreamDojoPipeline::for_gr1();
        let request = PipelineRequest {
            action: Some(ActionTensor {
                values: vec![0.1, -0.2, 0.0, 0.5, -0.3, 0.8, 0.2],
                previous_values: None,
                embodiment_type: EmbodimentType::Gr1,
                step_index: 0,
                recent_gripper_states: vec![],
            }),
            condition_frame: None,
            inference_params: Some(InferenceParams {
                guidance_scale: 7.5,
                num_steps: 50,
                num_conditional_frames: 2,
                resolution: [512, 512],
                seed: None,
                use_negative_prompt: false,
            }),
            latent_vector: Some(LatentVector {
                values: (0..32).map(|i| (i as f64 - 16.0) * 0.1).collect(),
                batch: vec![],
            }),
            predicted_chain: Some(
                (0..10)
                    .map(|i| PredictedFrame {
                        index: i,
                        pixel_mean: 128.0,
                        pixel_std: 30.0,
                        temporal_delta: Some(5.0),
                    })
                    .collect(),
            ),
        };
        let result = pipeline.evaluate(&request);
        assert!(result.is_safe);
        assert_eq!(result.guard_reports.len(), 4);
    }
}
