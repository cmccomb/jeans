//! Crossover operator abstractions for recombining chromosomes.

use crate::core::Chromosome;
use crate::ops::OperatorError;
use rand::RngCore;
use std::sync::Arc;

/// Produces new chromosomes by mixing genes from two parents.
///
/// # Examples
/// ```
/// use jeans::ops::CrossoverOperator;
/// use jeans::Chromosome;
/// use rand::thread_rng;
///
/// struct Swap;
///
/// impl CrossoverOperator for Swap {
///     fn crossover(&self, parent_a: &[f64], parent_b: &[f64], _rng: &mut dyn rand::RngCore) -> (Vec<f64>, Vec<f64>) {
///         (parent_b.to_vec(), parent_a.to_vec())
///     }
/// }
///
/// let operator = Swap;
/// let parent_a = Chromosome::new(vec![0.0, 1.0]);
/// let parent_b = Chromosome::new(vec![2.0, 3.0]);
/// let mut rng = thread_rng();
/// let (child_a, child_b) = operator.crossover_chromosomes(&parent_a, &parent_b, &mut rng);
/// assert_eq!(child_a.genes(), &[2.0, 3.0]);
/// assert_eq!(child_b.genes(), &[0.0, 1.0]);
/// ```
pub trait CrossoverOperator: Send + Sync {
    /// Applies crossover to parent slices and returns their offspring.
    fn crossover(
        &self,
        parent_a: &[f64],
        parent_b: &[f64],
        rng: &mut dyn RngCore,
    ) -> (Vec<f64>, Vec<f64>);

    /// Helper that operates directly on [`Chromosome`] values.
    fn crossover_chromosomes(
        &self,
        parent_a: &Chromosome,
        parent_b: &Chromosome,
        rng: &mut dyn RngCore,
    ) -> (Chromosome, Chromosome) {
        let (child_a, child_b) = self.crossover(parent_a.genes(), parent_b.genes(), rng);
        (Chromosome::new(child_a), Chromosome::new(child_b))
    }
}

impl<T: CrossoverOperator + ?Sized> CrossoverOperator for &T {
    fn crossover(
        &self,
        parent_a: &[f64],
        parent_b: &[f64],
        rng: &mut dyn RngCore,
    ) -> (Vec<f64>, Vec<f64>) {
        (**self).crossover(parent_a, parent_b, rng)
    }
}

impl<T: CrossoverOperator + ?Sized> CrossoverOperator for &mut T {
    fn crossover(
        &self,
        parent_a: &[f64],
        parent_b: &[f64],
        rng: &mut dyn RngCore,
    ) -> (Vec<f64>, Vec<f64>) {
        (**self).crossover(parent_a, parent_b, rng)
    }
}

impl<T: CrossoverOperator + ?Sized> CrossoverOperator for Box<T> {
    fn crossover(
        &self,
        parent_a: &[f64],
        parent_b: &[f64],
        rng: &mut dyn RngCore,
    ) -> (Vec<f64>, Vec<f64>) {
        (**self).crossover(parent_a, parent_b, rng)
    }
}

impl<T: CrossoverOperator + ?Sized> CrossoverOperator for Arc<T> {
    fn crossover(
        &self,
        parent_a: &[f64],
        parent_b: &[f64],
        rng: &mut dyn RngCore,
    ) -> (Vec<f64>, Vec<f64>) {
        (**self).crossover(parent_a, parent_b, rng)
    }
}

/// Simulated Binary Crossover (SBX) implementation following Deb & Agrawal.
#[derive(Debug, Clone)]
pub struct SimulatedBinaryCrossover {
    distribution_index: f64,
}

impl SimulatedBinaryCrossover {
    /// Creates a new SBX operator.
    ///
    /// # Errors
    /// Returns [`OperatorError::InvalidDistributionIndex`] when the provided
    /// distribution index is non-positive or not finite.
    pub fn new(distribution_index: f64) -> Result<Self, OperatorError> {
        if !(distribution_index.is_finite() && distribution_index > 0.0) {
            return Err(OperatorError::InvalidDistributionIndex {
                operator: "sbx",
                value: distribution_index,
            });
        }
        Ok(Self { distribution_index })
    }

    fn crossover_gene(&self, value_a: f64, value_b: f64, rng: &mut dyn RngCore) -> (f64, f64) {
        if (value_a - value_b).abs() < f64::EPSILON {
            return (value_a, value_b);
        }
        let u = random_unit(rng);
        let beta = if u <= 0.5 {
            (2.0 * u).powf(1.0 / (self.distribution_index + 1.0))
        } else {
            (2.0 * (1.0 - u)).powf(-1.0 / (self.distribution_index + 1.0))
        };
        let child1 = 0.5 * ((1.0 + beta) * value_a + (1.0 - beta) * value_b);
        let child2 = 0.5 * ((1.0 - beta) * value_a + (1.0 + beta) * value_b);
        (child1, child2)
    }
}

impl CrossoverOperator for SimulatedBinaryCrossover {
    fn crossover(
        &self,
        parent_a: &[f64],
        parent_b: &[f64],
        rng: &mut dyn RngCore,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut child_a = Vec::with_capacity(parent_a.len());
        let mut child_b = Vec::with_capacity(parent_b.len());
        for (&value_a, &value_b) in parent_a.iter().zip(parent_b.iter()) {
            let (gene_a, gene_b) = self.crossover_gene(value_a, value_b, rng);
            child_a.push(gene_a);
            child_b.push(gene_b);
        }
        (child_a, child_b)
    }
}

/// Blend crossover with configurable α parameter.
#[derive(Debug, Clone)]
pub struct BlendAlphaCrossover {
    alpha: f64,
}

impl BlendAlphaCrossover {
    /// Creates a new BLX-α operator.
    ///
    /// # Errors
    /// Returns [`OperatorError::InvalidParameter`] when `alpha` is negative or
    /// not finite.
    pub fn new(alpha: f64) -> Result<Self, OperatorError> {
        if !(alpha.is_finite() && alpha >= 0.0) {
            return Err(OperatorError::InvalidParameter {
                operator: "blx-alpha",
                parameter: "alpha",
                value: alpha,
            });
        }
        Ok(Self { alpha })
    }

    fn sample_gene(&self, value_a: f64, value_b: f64, rng: &mut dyn RngCore) -> f64 {
        let min = value_a.min(value_b);
        let max = value_a.max(value_b);
        let range = max - min;
        let lower = min - self.alpha * range;
        let upper = max + self.alpha * range;
        let roll = random_unit(rng);
        lower + roll * (upper - lower)
    }
}

impl CrossoverOperator for BlendAlphaCrossover {
    fn crossover(
        &self,
        parent_a: &[f64],
        parent_b: &[f64],
        rng: &mut dyn RngCore,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut child_a = Vec::with_capacity(parent_a.len());
        let mut child_b = Vec::with_capacity(parent_b.len());
        for (&value_a, &value_b) in parent_a.iter().zip(parent_b.iter()) {
            let gene_a = self.sample_gene(value_a, value_b, rng);
            let gene_b = self.sample_gene(value_a, value_b, rng);
            child_a.push(gene_a);
            child_b.push(gene_b);
        }
        (child_a, child_b)
    }
}

#[allow(clippy::cast_precision_loss)]
fn random_unit(rng: &mut dyn RngCore) -> f64 {
    let value = rng.next_u64() as f64;
    value / (u64::MAX as f64 + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    struct Swap;

    impl CrossoverOperator for Swap {
        fn crossover(
            &self,
            parent_a: &[f64],
            parent_b: &[f64],
            _rng: &mut dyn RngCore,
        ) -> (Vec<f64>, Vec<f64>) {
            (parent_b.to_vec(), parent_a.to_vec())
        }
    }

    #[test]
    fn crossover_helper_wraps_chromosomes() {
        let operator = Swap;
        let parent_a = Chromosome::new(vec![0.0, 1.0]);
        let parent_b = Chromosome::new(vec![2.0, 3.0]);
        let mut rng = thread_rng();
        let (child_a, child_b) = operator.crossover_chromosomes(&parent_a, &parent_b, &mut rng);
        assert_eq!(child_a.genes(), &[2.0, 3.0]);
        assert_eq!(child_b.genes(), &[0.0, 1.0]);
    }

    #[test]
    fn sbx_respects_identical_parents() {
        let operator = SimulatedBinaryCrossover::new(15.0).unwrap();
        let parent_a = [1.0, 2.0];
        let mut rng = thread_rng();
        let (child_a, child_b) = operator.crossover(&parent_a, &parent_a, &mut rng);
        assert_eq!(child_a, parent_a);
        assert_eq!(child_b, parent_a);
    }

    #[test]
    fn blx_alpha_samples_inflated_range() {
        let operator = BlendAlphaCrossover::new(0.5).unwrap();
        let parent_a = [0.0];
        let parent_b = [2.0];
        let mut rng = thread_rng();
        let (child_a, child_b) = operator.crossover(&parent_a, &parent_b, &mut rng);
        for value in child_a.into_iter().chain(child_b.into_iter()) {
            assert!((-1.0..=3.0).contains(&value));
        }
    }
}
