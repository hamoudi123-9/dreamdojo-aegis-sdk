use std::time::Instant;

use crate::traits::GuardEngine;
use crate::types::*;

/// WorldModelInputGuard validates condition frames (visual inputs) for
/// world model inference. SDK version provides basic dimension, all-zero,
/// uniform color, mean/std anomaly, and simple high-frequency checks.
pub struct WorldModelInputGuard {
    config: InputGuardConfig,
}

impl WorldModelInputGuard {
    pub fn new() -> Self {
        Self {
            config: InputGuardConfig::default(),
        }
    }

    pub fn with_config(config: InputGuardConfig) -> Self {
        Self { config }
    }

    fn compute_stats(&self, frame: &ConditionFrame) -> PixelStats {
        if frame.pixel_data.is_empty() {
            return PixelStats {
                mean: 0.0,
                std_dev: 0.0,
                min: 0,
                max: 0,
            };
        }

        let n = frame.pixel_data.len() as f64;
        let sum: f64 = frame.pixel_data.iter().map(|&p| p as f64).sum();
        let mean = sum / n;

        let variance: f64 = frame
            .pixel_data
            .iter()
            .map(|&p| {
                let d = p as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n;

        let min = frame.pixel_data.iter().copied().min().unwrap_or(0);
        let max = frame.pixel_data.iter().copied().max().unwrap_or(0);

        PixelStats {
            mean,
            std_dev: variance.sqrt(),
            min,
            max,
        }
    }

    fn check_dimension(&self, frame: &ConditionFrame) -> Vec<InputViolation> {
        let expected = self.config.expected_height * self.config.expected_width * self.config.expected_channels;
        if frame.pixel_data.len() != expected {
            vec![InputViolation {
                category: InputViolationCategory::DimensionMismatch,
                message: format!(
                    "Expected {}x{}x{} = {} pixels, got {}",
                    self.config.expected_height,
                    self.config.expected_width,
                    self.config.expected_channels,
                    expected,
                    frame.pixel_data.len()
                ),
                severity: 1.0,
                measured_value: frame.pixel_data.len() as f64,
                threshold: expected as f64,
            }]
        } else {
            vec![]
        }
    }

    fn check_all_zero(&self, frame: &ConditionFrame) -> Vec<InputViolation> {
        if !frame.pixel_data.is_empty() && frame.pixel_data.iter().all(|&p| p == 0) {
            vec![InputViolation {
                category: InputViolationCategory::AllZero,
                message: "All-zero frame detected (blank/corrupted input)".into(),
                severity: 0.9,
                measured_value: 0.0,
                threshold: 0.0,
            }]
        } else {
            vec![]
        }
    }

    fn check_uniform_color(&self, stats: &PixelStats) -> Vec<InputViolation> {
        if stats.std_dev < self.config.uniform_color_std_threshold && stats.std_dev >= 0.0 {
            vec![InputViolation {
                category: InputViolationCategory::UniformColor,
                message: format!(
                    "Near-uniform color detected (std_dev={:.2} < {:.2})",
                    stats.std_dev, self.config.uniform_color_std_threshold
                ),
                severity: 0.7,
                measured_value: stats.std_dev,
                threshold: self.config.uniform_color_std_threshold,
            }]
        } else {
            vec![]
        }
    }

    fn check_mean_anomaly(&self, stats: &PixelStats) -> Vec<InputViolation> {
        let mut violations = Vec::new();
        if stats.mean < self.config.min_mean {
            violations.push(InputViolation {
                category: InputViolationCategory::MeanAnomaly,
                message: format!(
                    "Pixel mean {:.2} below minimum {:.2} (too dark)",
                    stats.mean, self.config.min_mean
                ),
                severity: 0.5,
                measured_value: stats.mean,
                threshold: self.config.min_mean,
            });
        }
        if stats.mean > self.config.max_mean {
            violations.push(InputViolation {
                category: InputViolationCategory::MeanAnomaly,
                message: format!(
                    "Pixel mean {:.2} above maximum {:.2} (too bright)",
                    stats.mean, self.config.max_mean
                ),
                severity: 0.5,
                measured_value: stats.mean,
                threshold: self.config.max_mean,
            });
        }
        violations
    }

    fn check_std_anomaly(&self, stats: &PixelStats) -> Vec<InputViolation> {
        let mut violations = Vec::new();
        if stats.std_dev < self.config.min_std {
            // skip if already caught by uniform color
            return violations;
        }
        if stats.std_dev > self.config.max_std {
            violations.push(InputViolation {
                category: InputViolationCategory::StdAnomaly,
                message: format!(
                    "Pixel std_dev {:.2} exceeds maximum {:.2}",
                    stats.std_dev, self.config.max_std
                ),
                severity: 0.6,
                measured_value: stats.std_dev,
                threshold: self.config.max_std,
            });
        }
        violations
    }

    /// SDK basic high-frequency check: simple L2 norm of differences.
    /// Pro version uses 2nd-order finite differences for spectral analysis.
    fn check_high_frequency(&self, frame: &ConditionFrame) -> Vec<InputViolation> {
        if frame.pixel_data.len() < 3 {
            return vec![];
        }

        let n = frame.pixel_data.len() - 1;
        let energy: f64 = frame
            .pixel_data
            .windows(2)
            .map(|w| {
                let d = w[1] as f64 - w[0] as f64;
                d * d
            })
            .sum::<f64>()
            / n as f64;

        if energy > self.config.high_freq_threshold {
            vec![InputViolation {
                category: InputViolationCategory::HighFrequencyNoise,
                message: format!(
                    "High-frequency energy {:.2} exceeds threshold {:.2}",
                    energy, self.config.high_freq_threshold
                ),
                severity: 0.7,
                measured_value: energy,
                threshold: self.config.high_freq_threshold,
            }]
        } else {
            vec![]
        }
    }
}

impl Default for WorldModelInputGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl GuardEngine for WorldModelInputGuard {
    type Input = ConditionFrame;
    type Output = InputGuardResult;

    fn evaluate(&self, frame: &ConditionFrame) -> InputGuardResult {
        let start = Instant::now();
        let stats = self.compute_stats(frame);
        let mut violations = Vec::new();

        violations.extend(self.check_dimension(frame));

        // Only run content checks if dimension is correct
        if violations.is_empty() {
            violations.extend(self.check_all_zero(frame));
            violations.extend(self.check_uniform_color(&stats));
            violations.extend(self.check_mean_anomaly(&stats));
            violations.extend(self.check_std_anomaly(&stats));
            violations.extend(self.check_high_frequency(frame));
        }

        let messages: Vec<String> = violations.iter().map(|v| v.message.clone()).collect();
        let risk_score = violations
            .iter()
            .map(|v| v.severity)
            .fold(0.0_f64, f64::max);
        let is_safe = risk_score < self.config.block_threshold;

        InputGuardResult {
            is_safe,
            risk_score,
            violations,
            messages,
            pixel_stats: stats,
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(pixel_val: u8) -> ConditionFrame {
        ConditionFrame {
            pixel_data: vec![pixel_val; 480 * 640 * 3],
            height: 480,
            width: 640,
            channels: 3,
            frame_index: 0,
        }
    }

    #[test]
    fn test_valid_frame() {
        let guard = WorldModelInputGuard::new();
        // Generate varied pixel data
        let data: Vec<u8> = (0..480 * 640 * 3).map(|i| (i % 200 + 28) as u8).collect();
        let frame = ConditionFrame {
            pixel_data: data,
            height: 480,
            width: 640,
            channels: 3,
            frame_index: 0,
        };
        let result = guard.evaluate(&frame);
        assert!(result.is_safe);
    }

    #[test]
    fn test_all_zero() {
        let guard = WorldModelInputGuard::new();
        let result = guard.evaluate(&make_frame(0));
        assert!(result.violations.iter().any(|v| v.category == InputViolationCategory::AllZero));
    }

    #[test]
    fn test_uniform_color() {
        let guard = WorldModelInputGuard::new();
        let result = guard.evaluate(&make_frame(128));
        assert!(result.violations.iter().any(|v| v.category == InputViolationCategory::UniformColor));
    }

    #[test]
    fn test_dimension_mismatch() {
        let guard = WorldModelInputGuard::new();
        let frame = ConditionFrame {
            pixel_data: vec![128; 100],
            height: 480,
            width: 640,
            channels: 3,
            frame_index: 0,
        };
        let result = guard.evaluate(&frame);
        assert!(!result.is_safe);
    }
}
