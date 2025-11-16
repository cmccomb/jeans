//! Genetic operators and problem abstractions.
//!
//! This module groups the traits that describe how users interact with the
//! genetic algorithm core.  Each sub-module focuses on a particular aspect of
//! the optimization workflow so the implementations can stay lightweight and
//! single-purpose.

pub mod crossover;
pub mod mutation;
pub mod problem;
pub mod selection;

use crate::core::BoundsError;
use std::error::Error;
use std::fmt::{self, Display, Formatter};

pub use crossover::{BlendAlphaCrossover, CrossoverOperator, SimulatedBinaryCrossover};
pub use mutation::{GaussianMutation, MutationOperator, PolynomialMutation};
pub use problem::{
    AsyncProblem, MultiObjectiveProblem, Problem, ProblemBounds, ProblemError, ProblemResult,
};
pub use selection::SelectionOperator;

/// Errors emitted by operator constructors.
#[derive(Debug)]
pub enum OperatorError {
    /// Distribution index provided to SBX or polynomial mutation was invalid.
    InvalidDistributionIndex {
        /// Name of the operator reporting the error.
        operator: &'static str,
        /// Offending value.
        value: f64,
    },
    /// Probability parameter was outside `[0, 1]`.
    InvalidProbability {
        /// Name of the operator reporting the error.
        operator: &'static str,
        /// Offending value.
        value: f64,
    },
    /// Generic invalid parameter that does not fit other variants.
    InvalidParameter {
        /// Name of the operator reporting the error.
        operator: &'static str,
        /// Name of the parameter that failed validation.
        parameter: &'static str,
        /// Offending value.
        value: f64,
    },
    /// Wrapper around [`crate::core::BoundsError`].
    Bounds(BoundsError),
}

impl Display for OperatorError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDistributionIndex { operator, value } => {
                write!(
                    f,
                    "{operator} distribution index must be positive (received {value})"
                )
            }
            Self::InvalidProbability { operator, value } => {
                write!(
                    f,
                    "{operator} probability must be within [0, 1] (received {value})"
                )
            }
            Self::InvalidParameter {
                operator,
                parameter,
                value,
            } => {
                write!(
                    f,
                    "{operator} parameter {parameter} was invalid (received {value})"
                )
            }
            Self::Bounds(err) => write!(f, "{err}"),
        }
    }
}

impl Error for OperatorError {}

impl From<BoundsError> for OperatorError {
    fn from(err: BoundsError) -> Self {
        Self::Bounds(err)
    }
}
