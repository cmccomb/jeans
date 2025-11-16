//! Selection operator abstractions for real-coded genetic algorithms.

use crate::core::Chromosome;
use rand::RngCore;
use std::sync::Arc;

/// Selects parents from a population according to their fitness.
///
/// # Examples
/// ```
/// use jeans::ops::SelectionOperator;
/// use jeans::Chromosome;
/// use rand::thread_rng;
///
/// struct BestOnly;
///
/// impl SelectionOperator for BestOnly {
///     fn select_index(&self, fitness_values: &[f64], _rng: &mut dyn rand::RngCore) -> Option<usize> {
///         fitness_values
///             .iter()
///             .enumerate()
///             .max_by(|(_, a), (_, b)| a.total_cmp(b))
///             .map(|(idx, _)| idx)
///     }
/// }
///
/// let operator = BestOnly;
/// let population = vec![
///     Chromosome::new(vec![0.0]),
///     Chromosome::new(vec![1.0]),
/// ];
/// let fitness = vec![0.0, 10.0];
/// let mut rng = thread_rng();
/// let chromosome = operator.select(&population, &fitness, &mut rng).unwrap();
/// assert_eq!(chromosome.genes(), &[1.0]);
/// ```
pub trait SelectionOperator: Send + Sync {
    /// Returns the index of the chromosome to use as a parent.
    fn select_index(&self, fitness_values: &[f64], rng: &mut dyn RngCore) -> Option<usize>;

    /// Selects a [`Chromosome`] directly from the provided population.
    fn select<'a>(
        &self,
        population: &'a [Chromosome],
        fitness_values: &[f64],
        rng: &mut dyn RngCore,
    ) -> Option<&'a Chromosome> {
        if population.len() != fitness_values.len() {
            return None;
        }
        let idx = self.select_index(fitness_values, rng)?;
        population.get(idx)
    }

    /// Convenience helper that samples two parents.
    fn select_pair(&self, fitness_values: &[f64], rng: &mut dyn RngCore) -> Option<(usize, usize)> {
        let first = self.select_index(fitness_values, rng)?;
        let second = self.select_index(fitness_values, rng)?;
        Some((first, second))
    }
}

impl<T: SelectionOperator + ?Sized> SelectionOperator for &T {
    fn select_index(&self, fitness_values: &[f64], rng: &mut dyn RngCore) -> Option<usize> {
        (**self).select_index(fitness_values, rng)
    }
}

impl<T: SelectionOperator + ?Sized> SelectionOperator for &mut T {
    fn select_index(&self, fitness_values: &[f64], rng: &mut dyn RngCore) -> Option<usize> {
        (**self).select_index(fitness_values, rng)
    }
}

impl<T: SelectionOperator + ?Sized> SelectionOperator for Box<T> {
    fn select_index(&self, fitness_values: &[f64], rng: &mut dyn RngCore) -> Option<usize> {
        (**self).select_index(fitness_values, rng)
    }
}

impl<T: SelectionOperator + ?Sized> SelectionOperator for Arc<T> {
    fn select_index(&self, fitness_values: &[f64], rng: &mut dyn RngCore) -> Option<usize> {
        (**self).select_index(fitness_values, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    struct Deterministic;

    impl SelectionOperator for Deterministic {
        fn select_index(&self, _fitness_values: &[f64], _rng: &mut dyn RngCore) -> Option<usize> {
            Some(0)
        }
    }

    #[test]
    fn select_returns_matching_chromosome() {
        let operator = Deterministic;
        let population = vec![Chromosome::new(vec![1.0]), Chromosome::new(vec![2.0])];
        let fitness = vec![0.0, 1.0];
        let mut rng = thread_rng();
        let chromosome = operator.select(&population, &fitness, &mut rng).unwrap();
        assert_eq!(chromosome.genes(), &[1.0]);
    }

    #[test]
    fn select_pair_returns_two_indices() {
        let operator = Deterministic;
        let mut rng = thread_rng();
        let pair = operator.select_pair(&[1.0, 2.0], &mut rng).unwrap();
        assert_eq!(pair, (0, 0));
    }
}
