//! Mutation operator abstractions for perturbing chromosomes.

use crate::core::Chromosome;
use crate::ops::OperatorError;
use rand::RngCore;
use std::f64::consts::PI;
use std::sync::Arc;

/// Applies a mutation to a chromosome and returns a new candidate.
///
/// # Examples
/// ```
/// use jeans::ops::MutationOperator;
/// use jeans::Chromosome;
/// use rand::thread_rng;
///
/// struct AddOne;
///
/// impl MutationOperator for AddOne {
///     fn mutate(&self, parent: &[f64], _rng: &mut dyn rand::RngCore) -> Vec<f64> {
///         parent.iter().map(|value| value + 1.0).collect()
///     }
/// }
///
/// let operator = AddOne;
/// let parent = Chromosome::new(vec![0.0, 1.0]);
/// let mut rng = thread_rng();
/// let offspring = operator.mutate_chromosome(&parent, &mut rng);
/// assert_eq!(offspring.genes(), &[1.0, 2.0]);
/// ```
pub trait MutationOperator: Send + Sync {
    /// Mutates the provided chromosome slice.
    fn mutate(&self, parent: &[f64], rng: &mut dyn RngCore) -> Vec<f64>;

    /// Helper that mutates a [`Chromosome`] directly.
    fn mutate_chromosome(&self, chromosome: &Chromosome, rng: &mut dyn RngCore) -> Chromosome {
        Chromosome::new(self.mutate(chromosome.genes(), rng))
    }
}

impl<T: MutationOperator + ?Sized> MutationOperator for &T {
    fn mutate(&self, parent: &[f64], rng: &mut dyn RngCore) -> Vec<f64> {
        (**self).mutate(parent, rng)
    }
}

impl<T: MutationOperator + ?Sized> MutationOperator for &mut T {
    fn mutate(&self, parent: &[f64], rng: &mut dyn RngCore) -> Vec<f64> {
        (**self).mutate(parent, rng)
    }
}

impl<T: MutationOperator + ?Sized> MutationOperator for Box<T> {
    fn mutate(&self, parent: &[f64], rng: &mut dyn RngCore) -> Vec<f64> {
        (**self).mutate(parent, rng)
    }
}

impl<T: MutationOperator + ?Sized> MutationOperator for Arc<T> {
    fn mutate(&self, parent: &[f64], rng: &mut dyn RngCore) -> Vec<f64> {
        (**self).mutate(parent, rng)
    }
}

/// Polynomial mutation operator that respects problem bounds.
#[derive(Debug, Clone)]
pub struct PolynomialMutation {
    distribution_index: f64,
    probability: f64,
    lower_bounds: Vec<f64>,
    upper_bounds: Vec<f64>,
}

impl PolynomialMutation {
    /// Creates a new polynomial mutation operator.
    ///
    /// # Errors
    /// Returns [`OperatorError`] when the distribution index or probability is
    /// invalid, or when the bound vectors have mismatched lengths.
    pub fn new(
        lower_bounds: Vec<f64>,
        upper_bounds: Vec<f64>,
        distribution_index: f64,
        probability: f64,
    ) -> Result<Self, OperatorError> {
        if !(distribution_index.is_finite() && distribution_index > 0.0) {
            return Err(OperatorError::InvalidDistributionIndex {
                operator: "polynomial mutation",
                value: distribution_index,
            });
        }
        if !(probability.is_finite() && (0.0..=1.0).contains(&probability)) {
            return Err(OperatorError::InvalidProbability {
                operator: "polynomial mutation",
                value: probability,
            });
        }
        if lower_bounds.len() != upper_bounds.len() {
            return Err(OperatorError::from(
                crate::core::BoundsError::DimensionMismatch {
                    expected: lower_bounds.len(),
                    found: upper_bounds.len(),
                },
            ));
        }
        Ok(Self {
            distribution_index,
            probability,
            lower_bounds,
            upper_bounds,
        })
    }
}

impl MutationOperator for PolynomialMutation {
    fn mutate(&self, parent: &[f64], rng: &mut dyn RngCore) -> Vec<f64> {
        let mut child = parent.to_vec();
        for (idx, gene) in child.iter_mut().enumerate() {
            let roll = random_unit(rng);
            if roll > self.probability {
                continue;
            }
            let lower = self.lower_bounds[idx];
            let upper = self.upper_bounds[idx];
            let range = upper - lower;
            if range.abs() < f64::EPSILON {
                *gene = lower;
                continue;
            }
            let delta1 = (*gene - lower) / range;
            let delta2 = (upper - *gene) / range;
            let mut u = random_unit(rng);
            let mut delta_q = if u <= 0.5 {
                let term =
                    2.0 * u + (1.0 - 2.0 * u) * (1.0 - delta1).powf(self.distribution_index + 1.0);
                term.powf(1.0 / (self.distribution_index + 1.0)) - 1.0
            } else {
                u = 1.0 - u;
                let term =
                    2.0 * u + (1.0 - 2.0 * u) * (1.0 - delta2).powf(self.distribution_index + 1.0);
                1.0 - term.powf(1.0 / (self.distribution_index + 1.0))
            };
            if !delta_q.is_finite() {
                delta_q = 0.0;
            }
            *gene += delta_q * range;
            *gene = gene.clamp(lower, upper);
        }
        child
    }
}

/// Gaussian mutation with per-gene clamping.
#[derive(Debug, Clone)]
pub struct GaussianMutation {
    probability: f64,
    sigma: f64,
    lower_bounds: Vec<f64>,
    upper_bounds: Vec<f64>,
}

impl GaussianMutation {
    /// Creates a Gaussian mutation operator.
    ///
    /// # Errors
    /// Returns [`OperatorError`] when the bounds mismatch, probability lies
    /// outside `[0, 1]`, or `sigma` is non-positive.
    pub fn new(
        lower_bounds: Vec<f64>,
        upper_bounds: Vec<f64>,
        sigma: f64,
        probability: f64,
    ) -> Result<Self, OperatorError> {
        if !(sigma.is_finite() && sigma > 0.0) {
            return Err(OperatorError::InvalidParameter {
                operator: "gaussian mutation",
                parameter: "sigma",
                value: sigma,
            });
        }
        if !(probability.is_finite() && (0.0..=1.0).contains(&probability)) {
            return Err(OperatorError::InvalidProbability {
                operator: "gaussian mutation",
                value: probability,
            });
        }
        if lower_bounds.len() != upper_bounds.len() {
            return Err(OperatorError::from(
                crate::core::BoundsError::DimensionMismatch {
                    expected: lower_bounds.len(),
                    found: upper_bounds.len(),
                },
            ));
        }
        Ok(Self {
            probability,
            sigma,
            lower_bounds,
            upper_bounds,
        })
    }
}

impl MutationOperator for GaussianMutation {
    fn mutate(&self, parent: &[f64], rng: &mut dyn RngCore) -> Vec<f64> {
        let mut child = parent.to_vec();
        for (idx, gene) in child.iter_mut().enumerate() {
            if random_unit(rng) > self.probability {
                continue;
            }
            let perturbation = normal_sample(rng) * self.sigma;
            let lower = self.lower_bounds[idx];
            let upper = self.upper_bounds[idx];
            *gene = (*gene + perturbation).clamp(lower, upper);
        }
        child
    }
}

#[allow(clippy::cast_precision_loss)]
fn random_unit(rng: &mut dyn RngCore) -> f64 {
    let value = rng.next_u64() as f64;
    value / (u64::MAX as f64 + 1.0)
}

fn normal_sample(rng: &mut dyn RngCore) -> f64 {
    let u1 = loop {
        let sample = random_unit(rng);
        if sample > 0.0 {
            break sample;
        }
    };
    let u2 = random_unit(rng);
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    struct Offset(f64);

    impl MutationOperator for Offset {
        fn mutate(&self, parent: &[f64], _rng: &mut dyn RngCore) -> Vec<f64> {
            parent.iter().map(|value| value + self.0).collect()
        }
    }

    #[test]
    fn mutate_helper_wraps_chromosome() {
        let operator = Offset(0.5);
        let parent = Chromosome::new(vec![0.0, 1.0]);
        let mut rng = thread_rng();
        let offspring = operator.mutate_chromosome(&parent, &mut rng);
        assert_eq!(offspring.genes(), &[0.5, 1.5]);
    }

    #[test]
    fn polynomial_mutation_clamps_values() {
        let operator = PolynomialMutation::new(vec![0.0], vec![1.0], 20.0, 1.0).unwrap();
        let parent = [2.0];
        let mut rng = thread_rng();
        let child = operator.mutate(&parent, &mut rng);
        assert!(child[0] <= 1.0);
    }

    #[test]
    fn gaussian_mutation_stays_within_bounds() {
        let operator = GaussianMutation::new(vec![0.0], vec![1.0], 0.1, 1.0).unwrap();
        let parent = [0.5];
        let mut rng = thread_rng();
        let child = operator.mutate(&parent, &mut rng);
        assert!((0.0..=1.0).contains(&child[0]));
    }
}
