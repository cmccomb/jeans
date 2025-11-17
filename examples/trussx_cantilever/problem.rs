//! Example problem that sizes a three-member cantilever truss using `trussx`.
//!
//! The objective balances structural weight with tip deflection under a 15 kN
//! downward load. Candidate genes correspond to the cross-sectional area of
//! each member in square metres. The bounds are chosen to keep the properties
//! strictly positive while illustrating the effect of stiffer members.
//!
//! # Examples
//!
//! Build the problem and request a fitness evaluation for an arbitrary
//! candidate:
//!
//! ```ignore
//! use jeans::ops::{Problem, ProblemBounds};
//! use CantileverSizingProblem;
//!
//! let mut problem = CantileverSizingProblem::new();
//! assert_eq!(problem.dimensions(), 3);
//! let fitness = problem
//!     .evaluate(&[2.5e-4_f64, 2.5e-4_f64, 2.5e-4_f64])
//!     .expect("fitness evaluation should succeed");
//! println!("fitness: {fitness:.3}");
//! ```
//!
//! The module is intended for demonstration and testing; it is not published as
//! part of the crate API.

use jeans::ops::{Problem, ProblemBounds, ProblemError, ProblemResult};
use trussx::{force, point, Truss};

const STEEL_ELASTIC_MODULUS: f64 = 200.0e9;
const STEEL_DENSITY: f64 = 7_850.0;
const GRAVITY: f64 = 9.81;
const TIP_FORCE_NEWTONS: f64 = -15_000.0;
const TIP_DISPLACEMENT_LIMIT: f64 = 5.0e-4;
const DEFLECTION_WEIGHT: f64 = 50_000.0;
const LIMIT_PENALTY: f64 = 10_000_000.0;
const INVALID_PENALTY: f64 = 1_000_000_000.0;

/// Optimizes member areas for a three-member cantilever using `trussx` analysis.
pub struct CantileverSizingProblem {
    lower_bounds: Vec<f64>,
    upper_bounds: Vec<f64>,
    member_lengths: Vec<f64>,
}

impl CantileverSizingProblem {
    /// Create the sizing problem with precomputed member lengths and bounds.
    #[must_use]
    pub fn new() -> Self {
        let lower_bounds = vec![1.0e-4; 3];
        let upper_bounds = vec![8.0e-4; 3];
        let member_lengths = vec![2.0, 1.0, (5.0_f64).sqrt()];
        Self {
            lower_bounds,
            upper_bounds,
            member_lengths,
        }
    }

    fn invalid_penalty(&self) -> f64 {
        INVALID_PENALTY
    }

    fn analyze_candidate(&self, areas: &[f64]) -> Result<(f64, f64), ()> {
        let mut truss = Truss::new();
        let base = truss.add_joint(point(0.0, 0.0, 0.0));
        let roller = truss.add_joint(point(2.0, 0.0, 0.0));
        let tip = truss.add_joint(point(2.0, 1.0, 0.0));

        truss
            .set_support(base, [true, true, true])
            .map_err(|_| ())?;
        truss
            .set_support(roller, [true, true, true])
            .map_err(|_| ())?;
        truss
            .set_support(tip, [false, false, true])
            .map_err(|_| ())?;
        truss
            .set_load(tip, force(0.0, TIP_FORCE_NEWTONS, 0.0))
            .map_err(|_| ())?;

        let members = [
            truss.add_member(base, roller),
            truss.add_member(roller, tip),
            truss.add_member(base, tip),
        ];

        for (member, area) in members.iter().zip(areas.iter()) {
            truss
                .set_member_properties(*member, *area, STEEL_ELASTIC_MODULUS)
                .map_err(|_| ())?;
        }

        truss.evaluate().map_err(|_| ())?;
        let displacement = truss.joint_displacement(tip).ok_or(())?;
        let tip_deflection = displacement.y.abs();

        let weight = self.estimate_weight(areas);
        let deflection_cost = tip_deflection * DEFLECTION_WEIGHT;
        let limit_penalty = if tip_deflection > TIP_DISPLACEMENT_LIMIT {
            (tip_deflection - TIP_DISPLACEMENT_LIMIT) * LIMIT_PENALTY
        } else {
            0.0
        };

        Ok((weight + deflection_cost + limit_penalty, tip_deflection))
    }

    fn estimate_weight(&self, areas: &[f64]) -> f64 {
        areas
            .iter()
            .zip(self.member_lengths.iter())
            .map(|(area, length)| area * length * STEEL_DENSITY * GRAVITY)
            .sum()
    }

    /// Analyse a candidate and return both the fitness and tip deflection.
    #[must_use]
    pub fn analyze_with_deflection(&self, areas: &[f64]) -> Option<(f64, f64)> {
        self.analyze_candidate(areas).ok()
    }
}

impl ProblemBounds for CantileverSizingProblem {
    fn dimensions(&self) -> usize {
        self.lower_bounds.len()
    }

    fn lower_bounds(&self) -> &[f64] {
        &self.lower_bounds
    }

    fn upper_bounds(&self) -> &[f64] {
        &self.upper_bounds
    }
}

impl Problem for CantileverSizingProblem {
    fn evaluate(&mut self, genes: &[f64]) -> ProblemResult<f64> {
        if genes.len() != self.dimensions() {
            return Err(ProblemError::DimensionMismatch {
                expected: self.dimensions(),
                found: genes.len(),
            });
        }

        match self.analyze_candidate(genes) {
            Ok((fitness, _)) => Ok(fitness),
            Err(_) => Ok(self.invalid_penalty()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluates_to_finite_fitness() {
        // Arrange
        let mut problem = CantileverSizingProblem::new();
        let candidate = vec![2.5e-4; 3];

        // Act
        let fitness = problem.evaluate(&candidate).expect("fitness should exist");

        // Assert
        assert!(fitness.is_finite());
    }
}
