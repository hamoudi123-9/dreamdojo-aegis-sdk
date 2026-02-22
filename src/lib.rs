//! # AEGIS DreamDojo Guard SDK
//!
//! Safety validation framework for embodied AI world models.
//!
//! This SDK provides basic safety guards for robotic action spaces,
//! visual input validation, autoregressive chain monitoring,
//! latent space anomaly detection, and inference parameter validation.
//!
//! ## Quick Start
//!
//! ```rust
//! use aegis_dreamdojo_sdk::prelude::*;
//!
//! let guard = ActionSpaceGuard::new(EmbodimentType::Gr1);
//! let action = ActionTensor {
//!     values: vec![0.1, -0.2, 0.0, 0.5, -0.3, 0.8, 0.2],
//!     previous_values: None,
//!     embodiment_type: EmbodimentType::Gr1,
//!     step_index: 0,
//!     recent_gripper_states: vec![],
//! };
//! let result = guard.evaluate(&action);
//! assert!(result.is_safe);
//! ```

pub mod types;
pub mod traits;
pub mod action_guard;
pub mod input_guard;
pub mod chain_guard;
pub mod latent_guard;
pub mod guidance_guard;
pub mod pipeline;

/// Convenience re-exports for common usage.
pub mod prelude {
    pub use crate::types::*;
    pub use crate::traits::GuardEngine;
    pub use crate::action_guard::ActionSpaceGuard;
    pub use crate::input_guard::WorldModelInputGuard;
    pub use crate::chain_guard::AutoregressiveChainGuard;
    pub use crate::latent_guard::LatentSpaceGuard;
    pub use crate::guidance_guard::GuidanceGuard;
    pub use crate::pipeline::DreamDojoPipeline;
}
