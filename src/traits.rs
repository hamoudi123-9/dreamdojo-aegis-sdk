/// Core trait for all DreamDojo guard engines.
///
/// Each guard implements this trait with its specific input/output types.
/// The `evaluate` method performs safety validation and returns a result
/// containing risk scores, violations, and diagnostic messages.
pub trait GuardEngine {
    type Input: ?Sized;
    type Output;

    /// Evaluate the input and return a safety assessment.
    fn evaluate(&self, input: &Self::Input) -> Self::Output;
}
