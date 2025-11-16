//! Mutation operator abstractions for perturbing chromosomes.

use crate::core::Chromosome;
use rand::RngCore;
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
}
