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

pub use crossover::CrossoverOperator;
pub use mutation::MutationOperator;
pub use problem::{AsyncProblem, Problem, ProblemBounds, ProblemError, ProblemResult};
pub use selection::SelectionOperator;
