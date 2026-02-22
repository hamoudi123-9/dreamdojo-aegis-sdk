use aegis_dreamdojo_sdk::prelude::*;

fn main() {
    println!("=== AEGIS DreamDojo SDK: Full Pipeline Example ===\n");

    // Create pipeline for Fourier GR1
    let pipeline = DreamDojoPipeline::for_gr1();

    // Build a complete request with all guard inputs
    let request = PipelineRequest {
        action: Some(ActionTensor {
            values: vec![0.1, -0.2, 0.0, 0.5, -0.3, 0.8, 0.2],
            previous_values: None,
            embodiment_type: EmbodimentType::Gr1,
            step_index: 0,
            recent_gripper_states: vec![],
        }),
        condition_frame: None, // skip image validation in this example
        inference_params: Some(InferenceParams {
            guidance_scale: 7.5,
            num_steps: 50,
            num_conditional_frames: 2,
            resolution: [512, 512],
            seed: Some(42),
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
                    pixel_mean: 128.0 + i as f64 * 0.5,
                    pixel_std: 30.0,
                    temporal_delta: Some(3.0 + i as f64 * 0.2),
                })
                .collect(),
        ),
    };

    let result = pipeline.evaluate(&request);

    println!("Pipeline Result:");
    println!("  Overall Safe: {}", result.is_safe);
    println!("  Overall Risk: {:.4}", result.overall_risk);
    println!("  Risk Level:   {:?}", result.risk_level);
    println!("  Total Latency: {:.3}ms\n", result.latency_ms);

    println!("Guard Reports:");
    for report in &result.guard_reports {
        println!(
            "  {} — risk: {:.4}, passed: {}, violations: {}",
            report.guard_name, report.risk_score, report.passed, report.violation_count
        );
        for msg in &report.messages {
            println!("    - {}", msg);
        }
    }

    // Serialize result to JSON
    println!("\n--- JSON Output ---");
    let json = serde_json::to_string_pretty(&result).expect("serialize");
    // Print just the top-level summary (truncate for readability)
    for line in json.lines().take(15) {
        println!("{}", line);
    }
    println!("  ...");

    println!("\n=== Example Complete ===");
}
