use serde::{Deserialize, Serialize};

// ─── Embodiment Types ───

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbodimentType {
    Gr1,
    G1,
    Yam,
    AgiBot,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimBound {
    pub lower: f64,
    pub upper: f64,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbodimentProfile {
    pub embodiment_type: EmbodimentType,
    pub name: String,
    pub action_dim: usize,
    pub dim_bounds: Vec<DimBound>,
    pub max_velocity: Vec<f64>,
    pub gripper_indices: Vec<usize>,
    pub gripper_range: (f64, f64),
    pub max_gripper_flip_rate: f64,
}

impl EmbodimentProfile {
    /// GR1 humanoid — 7 action dimensions (SDK placeholder bounds)
    pub fn gr1() -> Self {
        let dim_names = [
            "base_x", "base_y", "base_yaw",
            "left_arm", "right_arm", "left_grip", "right_grip",
        ];
        Self {
            embodiment_type: EmbodimentType::Gr1,
            name: "Fourier GR1".into(),
            action_dim: 7,
            dim_bounds: dim_names
                .iter()
                .map(|n| DimBound {
                    lower: -std::f64::consts::PI,
                    upper: std::f64::consts::PI,
                    name: n.to_string(),
                })
                .collect(),
            max_velocity: vec![2.0; 7],
            gripper_indices: vec![5, 6],
            gripper_range: (0.0, 1.0),
            max_gripper_flip_rate: 0.5,
        }
    }

    /// Unitree G1 — 41 action dimensions (SDK placeholder bounds)
    pub fn g1() -> Self {
        Self {
            embodiment_type: EmbodimentType::G1,
            name: "Unitree G1".into(),
            action_dim: 41,
            dim_bounds: (0..41)
                .map(|i| DimBound {
                    lower: -std::f64::consts::PI,
                    upper: std::f64::consts::PI,
                    name: format!("joint_{}", i),
                })
                .collect(),
            max_velocity: vec![3.0; 41],
            gripper_indices: vec![20, 40],
            gripper_range: (0.0, 1.0),
            max_gripper_flip_rate: 0.5,
        }
    }

    /// Galaxea YAM — 14 action dimensions (SDK placeholder bounds)
    pub fn yam() -> Self {
        Self {
            embodiment_type: EmbodimentType::Yam,
            name: "Galaxea YAM".into(),
            action_dim: 14,
            dim_bounds: (0..14)
                .map(|i| DimBound {
                    lower: -std::f64::consts::PI,
                    upper: std::f64::consts::PI,
                    name: format!("joint_{}", i),
                })
                .collect(),
            max_velocity: vec![2.5; 14],
            gripper_indices: vec![6, 13],
            gripper_range: (0.0, 1.0),
            max_gripper_flip_rate: 0.5,
        }
    }

    /// AgiBot — 14 action dimensions (SDK placeholder bounds)
    pub fn agibot() -> Self {
        Self {
            embodiment_type: EmbodimentType::AgiBot,
            name: "AgiBot".into(),
            action_dim: 14,
            dim_bounds: (0..14)
                .map(|i| DimBound {
                    lower: -std::f64::consts::PI,
                    upper: std::f64::consts::PI,
                    name: format!("joint_{}", i),
                })
                .collect(),
            max_velocity: vec![2.5; 14],
            gripper_indices: vec![6, 13],
            gripper_range: (0.0, 1.0),
            max_gripper_flip_rate: 0.5,
        }
    }

    pub fn for_type(embodiment_type: EmbodimentType) -> Self {
        match embodiment_type {
            EmbodimentType::Gr1 => Self::gr1(),
            EmbodimentType::G1 => Self::g1(),
            EmbodimentType::Yam => Self::yam(),
            EmbodimentType::AgiBot => Self::agibot(),
            EmbodimentType::Custom => Self {
                embodiment_type: EmbodimentType::Custom,
                name: "Custom".into(),
                action_dim: 1,
                dim_bounds: vec![DimBound {
                    lower: -std::f64::consts::PI,
                    upper: std::f64::consts::PI,
                    name: "dim_0".into(),
                }],
                max_velocity: vec![2.0],
                gripper_indices: vec![],
                gripper_range: (0.0, 1.0),
                max_gripper_flip_rate: 0.5,
            },
        }
    }
}

// ─── Action Guard Types ───

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionTensor {
    pub values: Vec<f64>,
    #[serde(default)]
    pub previous_values: Option<Vec<f64>>,
    pub embodiment_type: EmbodimentType,
    #[serde(default)]
    pub step_index: usize,
    #[serde(default)]
    pub recent_gripper_states: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionViolationCategory {
    NanInf,
    DimensionMismatch,
    RangeViolation,
    VelocitySpike,
    ZeroAction,
    GripperRange,
    ScaleBoundary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionViolation {
    pub category: ActionViolationCategory,
    pub dimension: Option<usize>,
    pub message: String,
    pub severity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionGuardResult {
    pub is_safe: bool,
    pub risk_score: f64,
    pub violations: Vec<ActionViolation>,
    pub messages: Vec<String>,
    pub latency_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionGuardConfig {
    pub scale_boundary_multiplier: f64,
    pub allow_zero_action: bool,
    pub block_threshold: f64,
}

impl Default for ActionGuardConfig {
    fn default() -> Self {
        Self {
            scale_boundary_multiplier: 50.0, // SDK: relaxed (Pro: 20.0)
            allow_zero_action: false,
            block_threshold: 0.8,
        }
    }
}

// ─── Input Guard Types ───

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionFrame {
    pub pixel_data: Vec<u8>,
    #[serde(default = "default_height")]
    pub height: usize,
    #[serde(default = "default_width")]
    pub width: usize,
    #[serde(default = "default_channels")]
    pub channels: usize,
    #[serde(default)]
    pub frame_index: usize,
}

fn default_height() -> usize { 480 }
fn default_width() -> usize { 640 }
fn default_channels() -> usize { 3 }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputViolationCategory {
    DimensionMismatch,
    AllZero,
    UniformColor,
    MeanAnomaly,
    StdAnomaly,
    HighFrequencyNoise,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputViolation {
    pub category: InputViolationCategory,
    pub message: String,
    pub severity: f64,
    pub measured_value: f64,
    pub threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PixelStats {
    pub mean: f64,
    pub std_dev: f64,
    pub min: u8,
    pub max: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputGuardResult {
    pub is_safe: bool,
    pub risk_score: f64,
    pub violations: Vec<InputViolation>,
    pub messages: Vec<String>,
    pub pixel_stats: PixelStats,
    pub latency_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputGuardConfig {
    pub expected_height: usize,
    pub expected_width: usize,
    pub expected_channels: usize,
    pub min_mean: f64,
    pub max_mean: f64,
    pub min_std: f64,
    pub max_std: f64,
    pub high_freq_threshold: f64,
    pub uniform_color_std_threshold: f64,
    pub block_threshold: f64,
}

impl Default for InputGuardConfig {
    fn default() -> Self {
        Self {
            expected_height: 480,
            expected_width: 640,
            expected_channels: 3,
            min_mean: 10.0,
            max_mean: 245.0,
            min_std: 5.0,
            max_std: 120.0,
            high_freq_threshold: 2000.0, // SDK: relaxed (Pro: 500.0)
            uniform_color_std_threshold: 2.0,
            block_threshold: 0.8,
        }
    }
}

// ─── Chain Guard Types ───

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChainViolationCategory {
    TemporalDeltaSpike,
    DriftAccumulation,
    ChainLengthExceeded,
    EmptyChain,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedFrame {
    pub index: usize,
    pub pixel_mean: f64,
    pub pixel_std: f64,
    #[serde(default)]
    pub temporal_delta: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainViolation {
    pub category: ChainViolationCategory,
    pub frame_index: usize,
    pub message: String,
    pub severity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainStats {
    pub frame_count: usize,
    pub mean_delta: f64,
    pub max_delta: f64,
    pub drift_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainGuardResult {
    pub is_safe: bool,
    pub risk_score: f64,
    pub violations: Vec<ChainViolation>,
    pub messages: Vec<String>,
    pub chain_stats: ChainStats,
    pub latency_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainGuardConfig {
    pub max_temporal_delta: f64,
    pub max_chain_length: usize,
    pub drift_threshold: f64,
    pub block_threshold: f64,
}

impl Default for ChainGuardConfig {
    fn default() -> Self {
        Self {
            max_temporal_delta: 50.0,
            max_chain_length: 64,
            drift_threshold: 500.0, // SDK: relaxed (Pro: 100.0)
            block_threshold: 0.8,
        }
    }
}

// ─── Latent Guard Types ───

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LatentViolationCategory {
    NanInf,
    DimensionMismatch,
    NormAnomaly,
    ElementSigma,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentVector {
    pub values: Vec<f64>,
    #[serde(default)]
    pub batch: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentViolation {
    pub category: LatentViolationCategory,
    pub message: String,
    pub severity: f64,
    pub dimension: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentStats {
    pub l2_norm: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub max_abs: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentGuardResult {
    pub is_safe: bool,
    pub risk_score: f64,
    pub violations: Vec<LatentViolation>,
    pub messages: Vec<String>,
    pub latent_stats: LatentStats,
    pub latency_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentGuardConfig {
    pub expected_dim: usize,
    pub min_l2_norm: f64,
    pub max_l2_norm: f64,
    pub max_element_sigma: f64,
    pub block_threshold: f64,
}

impl Default for LatentGuardConfig {
    fn default() -> Self {
        Self {
            expected_dim: 32,
            min_l2_norm: 0.1,
            max_l2_norm: 50.0,
            max_element_sigma: 6.0, // SDK: relaxed (Pro: 4.0)
            block_threshold: 0.8,
        }
    }
}

// ─── Guidance Guard Types ───

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuidanceViolationCategory {
    GuidanceScale,
    ExtremeGuidance,
    StepCount,
    ConditionalFrames,
    Resolution,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceParams {
    pub guidance_scale: f64,
    pub num_steps: usize,
    pub num_conditional_frames: usize,
    pub resolution: [usize; 2],
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub use_negative_prompt: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuidanceViolation {
    pub category: GuidanceViolationCategory,
    pub message: String,
    pub severity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuidanceGuardResult {
    pub is_safe: bool,
    pub risk_score: f64,
    pub violations: Vec<GuidanceViolation>,
    pub messages: Vec<String>,
    pub latency_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuidanceGuardConfig {
    pub min_guidance: f64,
    pub max_guidance: f64,
    pub extreme_guidance: f64,
    pub min_steps: usize,
    pub max_steps: usize,
    pub allowed_conditional_frames: Vec<usize>,
    pub allowed_resolutions: Vec<[usize; 2]>,
    pub block_threshold: f64,
}

impl Default for GuidanceGuardConfig {
    fn default() -> Self {
        Self {
            min_guidance: 1.0,
            max_guidance: 15.0,
            extreme_guidance: 50.0, // SDK: relaxed (Pro: 20.0)
            min_steps: 10,
            max_steps: 100,
            allowed_conditional_frames: vec![1, 2, 4, 8],
            allowed_resolutions: vec![[256, 256], [512, 512], [640, 480], [1024, 1024]],
            block_threshold: 0.8,
        }
    }
}

// ─── Pipeline Types ───

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskLevel {
    Safe,
    Warning,
    Danger,
    Blocked,
}

impl RiskLevel {
    /// SDK uses relaxed cutoffs (Pro uses tighter: 0.3/0.5/0.8)
    pub fn from_score(score: f64) -> Self {
        if score < 0.5 {
            RiskLevel::Safe
        } else if score < 0.7 {
            RiskLevel::Warning
        } else if score < 0.9 {
            RiskLevel::Danger
        } else {
            RiskLevel::Blocked
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardReport {
    pub guard_name: String,
    pub risk_score: f64,
    pub passed: bool,
    pub violation_count: usize,
    pub messages: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineRequest {
    #[serde(default)]
    pub action: Option<ActionTensor>,
    #[serde(default)]
    pub condition_frame: Option<ConditionFrame>,
    #[serde(default)]
    pub inference_params: Option<InferenceParams>,
    #[serde(default)]
    pub latent_vector: Option<LatentVector>,
    #[serde(default)]
    pub predicted_chain: Option<Vec<PredictedFrame>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineResult {
    pub is_safe: bool,
    pub overall_risk: f64,
    pub risk_level: RiskLevel,
    pub guard_reports: Vec<GuardReport>,
    pub action_result: Option<ActionGuardResult>,
    pub input_result: Option<InputGuardResult>,
    pub guidance_result: Option<GuidanceGuardResult>,
    pub latent_result: Option<LatentGuardResult>,
    pub chain_result: Option<ChainGuardResult>,
    pub latency_ms: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub action_config: ActionGuardConfig,
    pub input_config: InputGuardConfig,
    pub guidance_config: GuidanceGuardConfig,
    pub latent_config: LatentGuardConfig,
    pub chain_config: ChainGuardConfig,
    pub early_exit: bool,
}

