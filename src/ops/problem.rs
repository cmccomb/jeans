//! Problem abstractions used by the operators.

use crate::core::Chromosome;
use async_trait::async_trait;
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::sync::Arc;

/// Convenience alias used by the problem traits.
pub type ProblemResult<T> = Result<T, ProblemError>;

/// Errors produced when a candidate solution is incompatible with a problem.
#[derive(Debug, Clone, PartialEq)]
pub enum ProblemError {
    /// The candidate contains the wrong number of decision variables.
    DimensionMismatch {
        /// Number of variables expected by the problem.
        expected: usize,
        /// Number of variables provided by the candidate solution.
        found: usize,
    },
    /// The lower and upper bounds do not align with the expected dimensionality.
    BoundsLengthMismatch {
        /// Expected dimensionality of the problem.
        expected: usize,
        /// Number of lower bounds provided.
        lower: usize,
        /// Number of upper bounds provided.
        upper: usize,
    },
}

impl Display for ProblemError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, found } => {
                write!(
                    f,
                    "candidate has {found} decision variables but problem expects {expected}"
                )
            }
            Self::BoundsLengthMismatch {
                expected,
                lower,
                upper,
            } => {
                write!(
                    f,
                    "bounds lengths ({lower}, {upper}) do not match problem dimensionality ({expected})"
                )
            }
        }
    }
}

impl Error for ProblemError {}

/// Shared domain metadata required by both synchronous and asynchronous problems.
pub trait ProblemBounds {
    /// Returns the number of decision variables handled by the problem.
    fn dimensions(&self) -> usize;

    /// Returns the lower bounds used to clamp candidate solutions.
    fn lower_bounds(&self) -> &[f64];

    /// Returns the upper bounds used to clamp candidate solutions.
    fn upper_bounds(&self) -> &[f64];

    /// Returns lower and upper bounds as a tuple for ergonomic destructuring.
    fn bounds(&self) -> (&[f64], &[f64]) {
        (self.lower_bounds(), self.upper_bounds())
    }

    /// Ensures that all bound arrays match the reported dimensionality.
    ///
    /// # Errors
    /// Returns [`ProblemError::BoundsLengthMismatch`] when the provided bound
    /// vectors do not align with [`Self::dimensions`].
    fn validate_bounds(&self) -> ProblemResult<()> {
        let expected = self.dimensions();
        let lower = self.lower_bounds().len();
        let upper = self.upper_bounds().len();
        if expected != lower || expected != upper || lower != upper {
            return Err(ProblemError::BoundsLengthMismatch {
                expected,
                lower,
                upper,
            });
        }
        Ok(())
    }

    /// Ensures that a candidate with the provided length is valid for the problem.
    ///
    /// # Errors
    /// Returns [`ProblemError::DimensionMismatch`] when the candidate contains
    /// the wrong number of decision variables.
    fn validate_candidate_length(&self, candidate_len: usize) -> ProblemResult<()> {
        let expected = self.dimensions();
        if candidate_len != expected {
            return Err(ProblemError::DimensionMismatch {
                expected,
                found: candidate_len,
            });
        }
        Ok(())
    }

    /// Clamps the provided candidate to the feasible domain described by the bounds.
    ///
    /// # Errors
    /// Propagates any [`ProblemError`] emitted by [`Self::validate_bounds`] or
    /// [`Self::validate_candidate_length`].
    fn clamp_to_domain(&self, genes: &mut [f64]) -> ProblemResult<()> {
        self.validate_bounds()?;
        self.validate_candidate_length(genes.len())?;
        for (value, (lower, upper)) in genes
            .iter_mut()
            .zip(self.lower_bounds().iter().zip(self.upper_bounds().iter()))
        {
            *value = value.clamp(*lower, *upper);
        }
        Ok(())
    }
}

impl<T: ProblemBounds + ?Sized> ProblemBounds for &T {
    fn dimensions(&self) -> usize {
        (**self).dimensions()
    }

    fn lower_bounds(&self) -> &[f64] {
        (**self).lower_bounds()
    }

    fn upper_bounds(&self) -> &[f64] {
        (**self).upper_bounds()
    }
}

impl<T: ProblemBounds + ?Sized> ProblemBounds for &mut T {
    fn dimensions(&self) -> usize {
        (**self).dimensions()
    }

    fn lower_bounds(&self) -> &[f64] {
        (**self).lower_bounds()
    }

    fn upper_bounds(&self) -> &[f64] {
        (**self).upper_bounds()
    }
}

impl<T: ProblemBounds + ?Sized> ProblemBounds for Box<T> {
    fn dimensions(&self) -> usize {
        (**self).dimensions()
    }

    fn lower_bounds(&self) -> &[f64] {
        (**self).lower_bounds()
    }

    fn upper_bounds(&self) -> &[f64] {
        (**self).upper_bounds()
    }
}

impl<T: ProblemBounds + ?Sized> ProblemBounds for Arc<T> {
    fn dimensions(&self) -> usize {
        (**self).dimensions()
    }

    fn lower_bounds(&self) -> &[f64] {
        (**self).lower_bounds()
    }

    fn upper_bounds(&self) -> &[f64] {
        (**self).upper_bounds()
    }
}

/// Trait implemented by synchronous problems.
///
/// # Examples
/// ```
/// use jeans::ops::{Problem, ProblemBounds, ProblemResult};
/// use jeans::Chromosome;
///
/// struct Sphere {
///     lower: [f64; 2],
///     upper: [f64; 2],
/// }
///
/// impl ProblemBounds for Sphere {
///     fn dimensions(&self) -> usize { 2 }
///     fn lower_bounds(&self) -> &[f64] { &self.lower }
///     fn upper_bounds(&self) -> &[f64] { &self.upper }
/// }
///
/// impl Problem for Sphere {
///     fn evaluate(&mut self, genes: &[f64]) -> ProblemResult<f64> {
///         Ok(genes.iter().map(|value| value * value).sum())
///     }
/// }
///
/// let mut problem = Sphere { lower: [-5.0, -5.0], upper: [5.0, 5.0] };
/// let chromosome = Chromosome::new(vec![1.0, 2.0]);
/// assert_eq!(problem.evaluate_chromosome(&chromosome).unwrap(), 5.0);
/// ```
pub trait Problem: ProblemBounds {
    /// Evaluates the fitness of the provided chromosome.
    ///
    /// # Errors
    /// Implementations may return [`ProblemError`] to describe domain issues.
    fn evaluate(&mut self, genes: &[f64]) -> ProblemResult<f64>;

    /// Helper that evaluates a [`Chromosome`] directly.
    ///
    /// # Errors
    /// Propagates any [`ProblemError`] reported by [`Problem::evaluate`].
    fn evaluate_chromosome(&mut self, chromosome: &Chromosome) -> ProblemResult<f64> {
        self.evaluate(chromosome.genes())
    }
}

impl<T: Problem + ?Sized> Problem for &mut T {
    fn evaluate(&mut self, genes: &[f64]) -> ProblemResult<f64> {
        (**self).evaluate(genes)
    }
}

impl<T: Problem + ?Sized> Problem for Box<T> {
    fn evaluate(&mut self, genes: &[f64]) -> ProblemResult<f64> {
        (**self).evaluate(genes)
    }
}

/// Trait implemented by asynchronous problems.
#[async_trait]
pub trait AsyncProblem: ProblemBounds + Send + Sync {
    /// Evaluates the fitness asynchronously.
    async fn evaluate_async(&self, genes: &[f64]) -> ProblemResult<f64>;

    /// Helper that evaluates a [`Chromosome`] asynchronously.
    async fn evaluate_chromosome_async(&self, chromosome: &Chromosome) -> ProblemResult<f64> {
        self.evaluate_async(chromosome.genes()).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    struct TestProblem {
        lower: Vec<f64>,
        upper: Vec<f64>,
    }

    impl TestProblem {
        fn new() -> Self {
            Self {
                lower: vec![-1.0, -1.0],
                upper: vec![1.0, 1.0],
            }
        }
    }

    impl ProblemBounds for TestProblem {
        fn dimensions(&self) -> usize {
            2
        }

        fn lower_bounds(&self) -> &[f64] {
            &self.lower
        }

        fn upper_bounds(&self) -> &[f64] {
            &self.upper
        }
    }

    impl Problem for TestProblem {
        fn evaluate(&mut self, genes: &[f64]) -> ProblemResult<f64> {
            Ok(genes.iter().copied().sum())
        }
    }

    struct AsyncTestProblem(TestProblem);

    impl ProblemBounds for AsyncTestProblem {
        fn dimensions(&self) -> usize {
            self.0.dimensions()
        }

        fn lower_bounds(&self) -> &[f64] {
            self.0.lower_bounds()
        }

        fn upper_bounds(&self) -> &[f64] {
            self.0.upper_bounds()
        }
    }

    #[async_trait]
    impl AsyncProblem for AsyncTestProblem {
        async fn evaluate_async(&self, genes: &[f64]) -> ProblemResult<f64> {
            Ok(genes.iter().copied().product())
        }
    }

    #[test]
    fn clamp_to_domain_bounds_values() {
        let problem = TestProblem::new();
        let mut genes = vec![-2.0, 2.0];
        problem.clamp_to_domain(&mut genes).unwrap();
        assert_eq!(genes, vec![-1.0, 1.0]);
    }

    #[test]
    fn evaluate_chromosome_helper() {
        let mut problem = TestProblem::new();
        let chromosome = Chromosome::new(vec![1.0, 1.5]);
        let score = problem.evaluate_chromosome(&chromosome).unwrap();
        assert!((score - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn async_chromosome_helper() {
        let problem = AsyncTestProblem(TestProblem::new());
        let chromosome = Chromosome::new(vec![2.0, 3.0]);
        let score = block_on(problem.evaluate_chromosome_async(&chromosome)).unwrap();
        assert!((score - 6.0).abs() < f64::EPSILON);
    }
}
