use std::time::Instant;

use crate::traits::GuardEngine;
use crate::types::*;

/// LatentSpaceGuard validates latent vectors from world model encoders.
/// SDK version provides NaN/Inf, dimension, and basic L2 norm checks.
/// Pro version includes Pearson sliding-window correlation analysis.
pub struct LatentSpaceGuard {
    config: LatentGuardConfig,
}

impl LatentSpaceGuard {
    pub fn new() -> Self {
        Self {
            config: LatentGuardConfig::default(),
        }
    }

    pub fn with_config(config: LatentGuardConfig) -> Self {
        Self { config }
    }

    fn check_nan_inf(&self, latent: &LatentVector) -> Vec<LatentViolation> {
        let mut violations = Vec::new();
        for (i, v) in latent.values.iter().enumerate() {
            if v.is_nan() || v.is_infinite() {
                violations.push(LatentViolation {
                    category: LatentViolationCategory::NanInf,
                    message: format!("Latent dimension {} contains NaN/Inf: {}", i, v),
                    severity: 1.0,
                    dimension: Some(i),
                });
            }
        }
        violations
    }

    fn check_dimension(&self, latent: &LatentVector) -> Vec<LatentViolation> {
        if latent.values.len() != self.config.expected_dim {
            vec![LatentViolation {
                category: LatentViolationCategory::DimensionMismatch,
                message: format!(
                    "Expected {} latent dimensions, got {}",
                    self.config.expected_dim,
                    latent.values.len()
                ),
                severity: 1.0,
                dimension: None,
            }]
        } else {
            vec![]
        }
    }

    fn compute_stats(&self, latent: &LatentVector) -> LatentStats {
        if latent.values.is_empty() {
            return LatentStats {
                l2_norm: 0.0,
                mean: 0.0,
                std_dev: 0.0,
                max_abs: 0.0,
            };
        }

        let n = latent.values.len() as f64;
        let l2_norm = latent.values.iter().map(|v| v * v).sum::<f64>().sqrt();
        let mean = latent.values.iter().sum::<f64>() / n;
        let variance = latent.values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let max_abs = latent
            .values
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);

        LatentStats {
            l2_norm,
            mean,
            std_dev: variance.sqrt(),
            max_abs,
        }
    }

    fn check_norm(&self, stats: &LatentStats) -> Vec<LatentViolation> {
        let mut violations = Vec::new();
        if stats.l2_norm < self.config.min_l2_norm {
            violations.push(LatentViolation {
                category: LatentViolationCategory::NormAnomaly,
                message: format!(
                    "L2 norm {:.4} below minimum {:.2} (collapsed representation)",
                    stats.l2_norm, self.config.min_l2_norm
                ),
                severity: 0.7,
                dimension: None,
            });
        }
        if stats.l2_norm > self.config.max_l2_norm {
            violations.push(LatentViolation {
                category: LatentViolationCategory::NormAnomaly,
                message: format!(
                    "L2 norm {:.4} above maximum {:.2} (exploding representation)",
                    stats.l2_norm, self.config.max_l2_norm
                ),
                severity: 0.8,
                dimension: None,
            });
        }
        violations
    }

    /// SDK basic element sigma check using max_abs relative to std_dev.
    fn check_element_sigma(&self, latent: &LatentVector, stats: &LatentStats) -> Vec<LatentViolation> {
        if stats.std_dev < 1e-10 {
            return vec![];
        }

        let mut violations = Vec::new();
        for (i, v) in latent.values.iter().enumerate() {
            let sigma = (v - stats.mean).abs() / stats.std_dev;
            if sigma > self.config.max_element_sigma {
                violations.push(LatentViolation {
                    category: LatentViolationCategory::ElementSigma,
                    message: format!(
                        "Latent dim {} value {:.4} is {:.1} sigma from mean (max: {:.1})",
                        i, v, sigma, self.config.max_element_sigma
                    ),
                    severity: 0.6,
                    dimension: Some(i),
                });
            }
        }
        violations
    }
}

impl Default for LatentSpaceGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl GuardEngine for LatentSpaceGuard {
    type Input = LatentVector;
    type Output = LatentGuardResult;

    fn evaluate(&self, latent: &LatentVector) -> LatentGuardResult {
        let start = Instant::now();
        let mut violations = Vec::new();

        violations.extend(self.check_nan_inf(latent));
        let dim_violations = self.check_dimension(latent);
        let dim_mismatch = !dim_violations.is_empty();
        violations.extend(dim_violations);

        let stats = self.compute_stats(latent);

        if !dim_mismatch {
            violations.extend(self.check_norm(&stats));
            violations.extend(self.check_element_sigma(latent, &stats));
        }

        let messages: Vec<String> = violations.iter().map(|v| v.message.clone()).collect();
        let risk_score = violations
            .iter()
            .map(|v| v.severity)
            .fold(0.0_f64, f64::max);
        let is_safe = risk_score < self.config.block_threshold;

        LatentGuardResult {
            is_safe,
            risk_score,
            violations,
            messages,
            latent_stats: stats,
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_latent() {
        let guard = LatentSpaceGuard::new();
        let latent = LatentVector {
            values: (0..32).map(|i| (i as f64 - 16.0) * 0.1).collect(),
            batch: vec![],
        };
        let result = guard.evaluate(&latent);
        assert!(result.is_safe);
    }

    #[test]
    fn test_nan_latent() {
        let guard = LatentSpaceGuard::new();
        let mut values = vec![0.5; 32];
        values[10] = f64::NAN;
        let latent = LatentVector { values, batch: vec![] };
        let result = guard.evaluate(&latent);
        assert!(!result.is_safe);
    }

    #[test]
    fn test_dim_mismatch() {
        let guard = LatentSpaceGuard::new();
        let latent = LatentVector { values: vec![0.1; 10], batch: vec![] };
        let result = guard.evaluate(&latent);
        assert!(!result.is_safe);
    }

    #[test]
    fn test_norm_exploding() {
        let guard = LatentSpaceGuard::new();
        let latent = LatentVector {
            values: vec![10.0; 32], // L2 norm = 10 * sqrt(32) ≈ 56.6 > 50.0
            batch: vec![],
        };
        let result = guard.evaluate(&latent);
        assert!(result.violations.iter().any(|v| v.category == LatentViolationCategory::NormAnomaly));
    }
}
