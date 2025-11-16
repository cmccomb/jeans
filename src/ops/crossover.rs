//! Crossover operator abstractions for recombining chromosomes.

use crate::core::Chromosome;
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
}
