use aegis_dreamdojo_sdk::prelude::*;

fn main() {
    println!("=== AEGIS DreamDojo SDK: Action Validation Example ===\n");

    // Create a guard for Fourier GR1 humanoid robot
    let guard = ActionSpaceGuard::new(EmbodimentType::Gr1);

    println!("Embodiment: {}", guard.profile().name);
    println!("Action dimensions: {}\n", guard.profile().action_dim);

    // Valid action
    let valid_action = ActionTensor {
        values: vec![0.1, -0.2, 0.0, 0.5, -0.3, 0.8, 0.2],
        previous_values: None,
        embodiment_type: EmbodimentType::Gr1,
        step_index: 0,
        recent_gripper_states: vec![],
    };

    let result = guard.evaluate(&valid_action);
    println!("[Valid Action]");
    println!("  Safe: {}", result.is_safe);
    println!("  Risk: {:.4}", result.risk_score);
    println!("  Violations: {}", result.violations.len());
    println!("  Latency: {:.3}ms\n", result.latency_ms);

    // Action with NaN (should be blocked)
    let nan_action = ActionTensor {
        values: vec![0.1, f64::NAN, 0.0, 0.5, -0.3, 0.8, 0.2],
        previous_values: None,
        embodiment_type: EmbodimentType::Gr1,
        step_index: 1,
        recent_gripper_states: vec![],
    };

    let result = guard.evaluate(&nan_action);
    println!("[NaN Action]");
    println!("  Safe: {}", result.is_safe);
    println!("  Risk: {:.4}", result.risk_score);
    for msg in &result.messages {
        println!("  - {}", msg);
    }
    println!();

    // Wrong dimension count
    let bad_dim_action = ActionTensor {
        values: vec![0.1, 0.2],
        previous_values: None,
        embodiment_type: EmbodimentType::Gr1,
        step_index: 2,
        recent_gripper_states: vec![],
    };

    let result = guard.evaluate(&bad_dim_action);
    println!("[Wrong Dimensions]");
    println!("  Safe: {}", result.is_safe);
    for msg in &result.messages {
        println!("  - {}", msg);
    }
    println!();

    // Custom config with stricter thresholds
    let strict_config = ActionGuardConfig {
        scale_boundary_multiplier: 10.0,
        allow_zero_action: false,
        block_threshold: 0.5,
    };
    let strict_guard = ActionSpaceGuard::with_config(EmbodimentType::Gr1, strict_config);

    let zero_action = ActionTensor {
        values: vec![0.0; 7],
        previous_values: None,
        embodiment_type: EmbodimentType::Gr1,
        step_index: 3,
        recent_gripper_states: vec![],
    };

    let result = strict_guard.evaluate(&zero_action);
    println!("[Zero Action with strict config (block_threshold=0.5)]");
    println!("  Safe: {}", result.is_safe);
    println!("  Risk: {:.4}", result.risk_score);
    for msg in &result.messages {
        println!("  - {}", msg);
    }

    println!("\n=== Example Complete ===");
}
