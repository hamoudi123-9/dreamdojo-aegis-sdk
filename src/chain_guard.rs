use std::time::Instant;

use crate::traits::GuardEngine;
use crate::types::*;

/// AutoregressiveChainGuard validates sequences of predicted frames for
/// temporal consistency. SDK version checks empty chains, chain length,
/// temporal delta spikes, and simple cumulative drift.
pub struct AutoregressiveChainGuard {
    config: ChainGuardConfig,
}

impl AutoregressiveChainGuard {
    pub fn new() -> Self {
        Self {
            config: ChainGuardConfig::default(),
        }
    }

    pub fn with_config(config: ChainGuardConfig) -> Self {
        Self { config }
    }

    fn check_empty(&self, frames: &[PredictedFrame]) -> Vec<ChainViolation> {
        if frames.is_empty() {
            vec![ChainViolation {
                category: ChainViolationCategory::EmptyChain,
                frame_index: 0,
                message: "Empty prediction chain".into(),
                severity: 1.0,
            }]
        } else {
            vec![]
        }
    }

    fn check_length(&self, frames: &[PredictedFrame]) -> Vec<ChainViolation> {
        if frames.len() > self.config.max_chain_length {
            vec![ChainViolation {
                category: ChainViolationCategory::ChainLengthExceeded,
                frame_index: frames.len() - 1,
                message: format!(
                    "Chain length {} exceeds maximum {}",
                    frames.len(),
                    self.config.max_chain_length
                ),
                severity: 0.8,
            }]
        } else {
            vec![]
        }
    }

    fn check_deltas(&self, frames: &[PredictedFrame]) -> (Vec<ChainViolation>, Vec<f64>) {
        let mut violations = Vec::new();
        let mut deltas = Vec::new();

        for frame in frames {
            if let Some(delta) = frame.temporal_delta {
                deltas.push(delta);
                if delta > self.config.max_temporal_delta {
                    violations.push(ChainViolation {
                        category: ChainViolationCategory::TemporalDeltaSpike,
                        frame_index: frame.index,
                        message: format!(
                            "Frame {} temporal delta {:.2} exceeds max {:.2}",
                            frame.index, delta, self.config.max_temporal_delta
                        ),
                        severity: 0.7,
                    });
                }
            }
        }

        (violations, deltas)
    }

    /// SDK drift check: simple cumulative sum of deltas.
    /// Pro version uses Welford online algorithm for streaming variance.
    fn check_drift(&self, deltas: &[f64]) -> Vec<ChainViolation> {
        if deltas.is_empty() {
            return vec![];
        }

        let cumulative: f64 = deltas.iter().sum();
        if cumulative > self.config.drift_threshold {
            vec![ChainViolation {
                category: ChainViolationCategory::DriftAccumulation,
                frame_index: deltas.len() - 1,
                message: format!(
                    "Cumulative drift {:.2} exceeds threshold {:.2}",
                    cumulative, self.config.drift_threshold
                ),
                severity: 0.8,
            }]
        } else {
            vec![]
        }
    }

    fn compute_stats(&self, frames: &[PredictedFrame], deltas: &[f64]) -> ChainStats {
        let mean_delta = if deltas.is_empty() {
            0.0
        } else {
            deltas.iter().sum::<f64>() / deltas.len() as f64
        };

        let max_delta = deltas.iter().copied().fold(0.0_f64, f64::max);
        let drift_score: f64 = deltas.iter().sum();

        ChainStats {
            frame_count: frames.len(),
            mean_delta,
            max_delta,
            drift_score,
        }
    }
}

impl Default for AutoregressiveChainGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl GuardEngine for AutoregressiveChainGuard {
    type Input = [PredictedFrame];
    type Output = ChainGuardResult;

    fn evaluate(&self, frames: &[PredictedFrame]) -> ChainGuardResult {
        let start = Instant::now();
        let mut violations = Vec::new();

        violations.extend(self.check_empty(frames));
        if !violations.is_empty() {
            return ChainGuardResult {
                is_safe: false,
                risk_score: 1.0,
                violations: violations.clone(),
                messages: violations.iter().map(|v| v.message.clone()).collect(),
                chain_stats: ChainStats {
                    frame_count: 0,
                    mean_delta: 0.0,
                    max_delta: 0.0,
                    drift_score: 0.0,
                },
                latency_ms: start.elapsed().as_secs_f64() * 1000.0,
            };
        }

        violations.extend(self.check_length(frames));
        let (delta_violations, deltas) = self.check_deltas(frames);
        violations.extend(delta_violations);
        violations.extend(self.check_drift(&deltas));

        let stats = self.compute_stats(frames, &deltas);
        let messages: Vec<String> = violations.iter().map(|v| v.message.clone()).collect();
        let risk_score = violations
            .iter()
            .map(|v| v.severity)
            .fold(0.0_f64, f64::max);
        let is_safe = risk_score < self.config.block_threshold;

        ChainGuardResult {
            is_safe,
            risk_score,
            violations,
            messages,
            chain_stats: stats,
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_chain() {
        let guard = AutoregressiveChainGuard::new();
        let frames: Vec<PredictedFrame> = (0..10)
            .map(|i| PredictedFrame {
                index: i,
                pixel_mean: 128.0,
                pixel_std: 30.0,
                temporal_delta: Some(5.0),
            })
            .collect();
        let result = guard.evaluate(&frames);
        assert!(result.is_safe);
    }

    #[test]
    fn test_empty_chain() {
        let guard = AutoregressiveChainGuard::new();
        let result = guard.evaluate(&[]);
        assert!(!result.is_safe);
        assert_eq!(result.violations[0].category, ChainViolationCategory::EmptyChain);
    }

    #[test]
    fn test_temporal_spike() {
        let guard = AutoregressiveChainGuard::new();
        let frames = vec![
            PredictedFrame { index: 0, pixel_mean: 128.0, pixel_std: 30.0, temporal_delta: Some(5.0) },
            PredictedFrame { index: 1, pixel_mean: 128.0, pixel_std: 30.0, temporal_delta: Some(100.0) },
        ];
        let result = guard.evaluate(&frames);
        assert!(result.violations.iter().any(|v| v.category == ChainViolationCategory::TemporalDeltaSpike));
    }
}
