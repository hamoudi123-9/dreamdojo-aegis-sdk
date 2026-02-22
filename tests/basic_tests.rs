use aegis_dreamdojo_sdk::prelude::*;

// ─── Action Guard ───

#[test]
fn action_guard_valid_gr1() {
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
    assert_eq!(result.violations.len(), 0);
}

#[test]
fn action_guard_nan_blocked() {
    let guard = ActionSpaceGuard::new(EmbodimentType::Gr1);
    let action = ActionTensor {
        values: vec![f64::NAN; 7],
        previous_values: None,
        embodiment_type: EmbodimentType::Gr1,
        step_index: 0,
        recent_gripper_states: vec![],
    };
    let result = guard.evaluate(&action);
    assert!(!result.is_safe);
    assert!(result.risk_score >= 0.8);
}

#[test]
fn action_guard_dimension_mismatch() {
    let guard = ActionSpaceGuard::new(EmbodimentType::G1);
    let action = ActionTensor {
        values: vec![0.1; 7], // G1 expects 41
        previous_values: None,
        embodiment_type: EmbodimentType::G1,
        step_index: 0,
        recent_gripper_states: vec![],
    };
    let result = guard.evaluate(&action);
    assert!(!result.is_safe);
}

#[test]
fn action_guard_velocity_spike() {
    let guard = ActionSpaceGuard::new(EmbodimentType::Gr1);
    let action = ActionTensor {
        values: vec![0.1, -0.2, 0.0, 0.5, -0.3, 0.8, 0.2],
        previous_values: Some(vec![0.1, -0.2, 0.0, 0.5, -0.3, 0.8, -3.0]), // big jump on dim 6
        embodiment_type: EmbodimentType::Gr1,
        step_index: 1,
        recent_gripper_states: vec![],
    };
    let result = guard.evaluate(&action);
    assert!(result.violations.iter().any(|v| v.category == ActionViolationCategory::VelocitySpike));
}

// ─── Input Guard ───

#[test]
fn input_guard_valid_frame() {
    let guard = WorldModelInputGuard::new();
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
fn input_guard_all_zero() {
    let guard = WorldModelInputGuard::new();
    let frame = ConditionFrame {
        pixel_data: vec![0; 480 * 640 * 3],
        height: 480,
        width: 640,
        channels: 3,
        frame_index: 0,
    };
    let result = guard.evaluate(&frame);
    assert!(result.violations.iter().any(|v| v.category == InputViolationCategory::AllZero));
}

// ─── Chain Guard ───

#[test]
fn chain_guard_valid() {
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
fn chain_guard_empty() {
    let guard = AutoregressiveChainGuard::new();
    let result = guard.evaluate(&[]);
    assert!(!result.is_safe);
}

// ─── Latent Guard ───

#[test]
fn latent_guard_valid() {
    let guard = LatentSpaceGuard::new();
    let latent = LatentVector {
        values: (0..32).map(|i| (i as f64 - 16.0) * 0.1).collect(),
        batch: vec![],
    };
    let result = guard.evaluate(&latent);
    assert!(result.is_safe);
}

#[test]
fn latent_guard_nan() {
    let guard = LatentSpaceGuard::new();
    let mut values = vec![0.5; 32];
    values[0] = f64::INFINITY;
    let latent = LatentVector { values, batch: vec![] };
    let result = guard.evaluate(&latent);
    assert!(!result.is_safe);
}

// ─── Guidance Guard ───

#[test]
fn guidance_guard_valid() {
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
fn guidance_guard_extreme() {
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

// ─── Pipeline ───

#[test]
fn pipeline_full_safe() {
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
            (0..8)
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
    assert_eq!(result.risk_level, RiskLevel::Safe);
    assert_eq!(result.guard_reports.len(), 4);
}

#[test]
fn pipeline_nan_blocks() {
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

// ─── Serialization ───

#[test]
fn types_serialize_roundtrip() {
    let result = ActionGuardResult {
        is_safe: true,
        risk_score: 0.0,
        violations: vec![],
        messages: vec![],
        latency_ms: 0.01,
    };
    let json = serde_json::to_string(&result).unwrap();
    let parsed: ActionGuardResult = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.is_safe, result.is_safe);
}

#[test]
fn embodiment_profiles_correct_dims() {
    assert_eq!(EmbodimentProfile::gr1().action_dim, 7);
    assert_eq!(EmbodimentProfile::g1().action_dim, 41);
    assert_eq!(EmbodimentProfile::yam().action_dim, 14);
    assert_eq!(EmbodimentProfile::agibot().action_dim, 14);
}
