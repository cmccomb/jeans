//! Core genetic algorithm primitives.
//!
//! This module provides building blocks that power the higher-level
//! optimization utilities exposed by the crate. The types are intentionally
//! lightweight and focused on correctness so they can be re-used in tests or
//! in bespoke experimentation.

use rand::distributions::{Distribution, Uniform};
use rand::seq::SliceRandom;
use rand::Rng;
use std::cmp::Ordering;
use std::convert::TryFrom;

/// Scalar type used to represent a single gene value.
///
/// # Examples
/// ```
/// use jeans::Gene;
/// let gene: Gene = 1.0;
/// assert_eq!(gene, 1.0);
/// ```
pub type Gene = f64;
/// Ordered collection of genes that defines a candidate solution.
///
/// # Examples
/// ```
/// use jeans::Chromosome;
/// let chromosome = Chromosome::new(vec![0.0, 1.0]);
/// assert_eq!(chromosome.len(), 2);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Chromosome {
    genes: Vec<Gene>,
}

impl Chromosome {
    /// Creates a chromosome from raw genes.
    ///
    /// # Examples
    /// ```
    /// use jeans::Chromosome;
    /// let chromosome = Chromosome::new(vec![0.0, 1.0]);
    /// assert_eq!(chromosome.len(), 2);
    /// ```
    #[must_use]
    pub fn new(genes: Vec<Gene>) -> Self {
        Self { genes }
    }

    /// Returns the number of genes stored in the chromosome.
    ///
    /// # Examples
    /// ```
    /// use jeans::Chromosome;
    /// let chromosome = Chromosome::new(vec![0.0, 1.0, 2.0]);
    /// assert_eq!(chromosome.len(), 3);
    /// ```
    #[must_use]
    pub fn len(&self) -> usize {
        self.genes.len()
    }

    /// Indicates whether the chromosome has zero genes.
    ///
    /// # Examples
    /// ```
    /// use jeans::Chromosome;
    /// let chromosome = Chromosome::new(vec![]);
    /// assert!(chromosome.is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.genes.is_empty()
    }

    /// Returns a shared slice with all genes.
    ///
    /// # Examples
    /// ```
    /// use jeans::Chromosome;
    /// let chromosome = Chromosome::new(vec![0.25, 0.5]);
    /// assert_eq!(chromosome.genes(), &[0.25, 0.5]);
    /// ```
    #[must_use]
    pub fn genes(&self) -> &[Gene] {
        &self.genes
    }

    /// Returns an iterator over the genes.
    ///
    /// # Examples
    /// ```
    /// use jeans::Chromosome;
    /// let chromosome = Chromosome::new(vec![1.0, 2.0]);
    /// assert_eq!(chromosome.iter().copied().collect::<Vec<_>>(), vec![1.0, 2.0]);
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = &Gene> {
        self.genes.iter()
    }

    /// Returns a mutable iterator over the genes.
    ///
    /// # Examples
    /// ```
    /// use jeans::Chromosome;
    /// let mut chromosome = Chromosome::new(vec![0.0, 1.0]);
    /// chromosome.iter_mut().for_each(|gene| *gene += 1.0);
    /// assert_eq!(chromosome.genes(), &[1.0, 2.0]);
    /// ```
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Gene> {
        self.genes.iter_mut()
    }

    /// Generates a random chromosome within the provided bounds.
    ///
    /// # Examples
    /// ```
    /// use jeans::{Chromosome, Gene};
    /// use rand::thread_rng;
    ///
    /// let lower: [Gene; 2] = [-1.0, 0.0];
    /// let upper: [Gene; 2] = [1.0, 0.5];
    /// let chromosome = Chromosome::random_with_bounds(&lower, &upper, &mut thread_rng()).unwrap();
    /// assert_eq!(chromosome.len(), 2);
    /// ```
    ///
    /// # Errors
    /// Returns [`BoundsError`] when the bounds have mismatched lengths or when
    /// any lower bound exceeds the corresponding upper bound.
    pub fn random_with_bounds(
        lower_bounds: &[Gene],
        upper_bounds: &[Gene],
        rng: &mut impl Rng,
    ) -> Result<Self, BoundsError> {
        validate_bounds(lower_bounds, upper_bounds)?;
        let mut genes = Vec::with_capacity(lower_bounds.len());
        for (idx, (lower, upper)) in lower_bounds.iter().zip(upper_bounds.iter()).enumerate() {
            let sampler = Uniform::new_inclusive(*lower, *upper);
            let value = sampler.sample(rng).min(*upper).max(*lower);
            if !value.is_finite() {
                return Err(BoundsError::InvalidRange {
                    dimension: idx,
                    lower: *lower,
                    upper: *upper,
                });
            }
            genes.push(value);
        }
        Ok(Self { genes })
    }

    pub(crate) fn random_from_settings(
        settings: &crate::Settings,
        rng: &mut impl Rng,
    ) -> Result<Self, BoundsError> {
        if settings.lower_bound.len() != settings.number_of_dimensions as usize
            || settings.upper_bound.len() != settings.number_of_dimensions as usize
        {
            return Err(BoundsError::DimensionMismatch {
                expected: settings.number_of_dimensions as usize,
                found: settings.lower_bound.len().max(settings.upper_bound.len()),
            });
        }
        Chromosome::random_with_bounds(&settings.lower_bound, &settings.upper_bound, rng)
    }
}
/// Representation of a solution that contains both a chromosome and a fitness
/// value.
///
/// # Examples
/// ```
/// use jeans::{Chromosome, Individual};
/// let individual = Individual::from_parts(Chromosome::new(vec![0.0]), 0.0);
/// assert_eq!(individual.fitness(), 0.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Individual {
    chromosome: Chromosome,
    fitness: Gene,
}

impl Individual {
    /// Creates a new individual from a chromosome and a provided fitness.
    ///
    /// # Examples
    /// ```
    /// use jeans::{Chromosome, Individual};
    /// let chromosome = Chromosome::new(vec![0.0, 1.0]);
    /// let individual = Individual::from_parts(chromosome, 42.0);
    /// assert_eq!(individual.fitness(), 42.0);
    /// ```
    #[must_use]
    pub fn from_parts(chromosome: Chromosome, fitness: Gene) -> Self {
        Self {
            chromosome,
            fitness,
        }
    }

    /// Creates a new individual with fitness evaluated through a callback.
    ///
    /// # Examples
    /// ```
    /// use jeans::{Chromosome, Individual};
    /// let chromosome = Chromosome::new(vec![2.0, 2.0]);
    /// let individual = Individual::evaluate_with(chromosome, |genes| genes.iter().copied().sum());
    /// assert_eq!(individual.fitness(), 4.0);
    /// ```
    pub fn evaluate_with<F>(chromosome: Chromosome, mut fitness_fn: F) -> Self
    where
        F: FnMut(&[Gene]) -> Gene,
    {
        let fitness = fitness_fn(chromosome.genes());
        Self {
            chromosome,
            fitness,
        }
    }

    /// Returns the inner chromosome.
    ///
    /// # Examples
    /// ```
    /// use jeans::{Chromosome, Individual};
    /// let chromosome = Chromosome::new(vec![0.0]);
    /// let individual = Individual::from_parts(chromosome.clone(), 0.0);
    /// assert_eq!(individual.chromosome().genes(), chromosome.genes());
    /// ```
    #[must_use]
    pub fn chromosome(&self) -> &Chromosome {
        &self.chromosome
    }

    /// Returns a mutable reference to the chromosome.
    ///
    /// # Examples
    /// ```
    /// use jeans::{Chromosome, Individual};
    /// let mut individual = Individual::from_parts(Chromosome::new(vec![0.0]), 0.0);
    /// individual.chromosome_mut().iter_mut().for_each(|gene| *gene += 1.0);
    /// assert_eq!(individual.chromosome().genes(), &[1.0]);
    /// ```
    pub fn chromosome_mut(&mut self) -> &mut Chromosome {
        &mut self.chromosome
    }

    /// Returns the current fitness value.
    ///
    /// # Examples
    /// ```
    /// use jeans::{Chromosome, Individual};
    /// let individual = Individual::from_parts(Chromosome::new(vec![0.0]), 1.5);
    /// assert_eq!(individual.fitness(), 1.5);
    /// ```
    #[must_use]
    pub fn fitness(&self) -> Gene {
        self.fitness
    }

    /// Updates the individual's fitness.
    ///
    /// # Examples
    /// ```
    /// use jeans::{Chromosome, Individual};
    /// let mut individual = Individual::from_parts(Chromosome::new(vec![0.0]), 0.0);
    /// individual.set_fitness(2.5);
    /// assert_eq!(individual.fitness(), 2.5);
    /// ```
    pub fn set_fitness(&mut self, fitness: Gene) {
        self.fitness = fitness;
    }

    /// Generates a random individual based on settings.
    ///
    /// # Examples
    /// ```
    /// use jeans::{Individual, Settings};
    /// let mut settings = Settings::default();
    /// let individual = Individual::new(&mut settings);
    /// assert_eq!(individual.chromosome().len() as u32, settings.number_of_dimensions);
    /// ```
    ///
    /// # Panics
    /// Panics when the settings contain inconsistent bounds that prevent
    /// random initialization.
    #[must_use]
    pub fn new(settings: &mut crate::Settings) -> Self {
        Self::try_new(settings).expect("settings bounds are valid")
    }

    /// Attempts to create an individual from the provided [`Settings`](crate::Settings).
    ///
    /// # Examples
    /// ```
    /// use jeans::Individual;
    /// let mut settings = jeans::Settings::default();
    /// let individual = Individual::try_new(&mut settings).unwrap();
    /// assert_eq!(individual.chromosome().len() as u32, settings.number_of_dimensions);
    /// ```
    ///
    /// # Errors
    /// Returns [`BoundsError`] when the bounds stored in `settings` are
    /// inconsistent with `number_of_dimensions`.
    pub fn try_new(settings: &mut crate::Settings) -> Result<Self, BoundsError> {
        let mut rng = rand::thread_rng();
        let chromosome = Chromosome::random_from_settings(settings, &mut rng)?;
        let genes = chromosome.genes().to_vec();
        let fitness = (settings.fitness_function)(genes);
        Ok(Self {
            chromosome,
            fitness,
        })
    }

    /// Returns a mutated version of the individual. Mutation logic is delegated
    /// to the provided closure, which can be deterministic.
    ///
    /// # Examples
    /// ```
    /// use jeans::{Chromosome, Individual};
    /// let individual = Individual::from_parts(Chromosome::new(vec![0.0]), 0.0);
    /// let mutated = individual.mutate_with(|chromosome| chromosome.clone());
    /// assert_eq!(mutated.chromosome().genes(), &[0.0]);
    /// ```
    #[must_use]
    pub fn mutate_with<F>(&self, mutator: F) -> Self
    where
        F: FnOnce(&Chromosome) -> Chromosome,
    {
        Self {
            chromosome: mutator(&self.chromosome),
            fitness: self.fitness,
        }
    }

    /// Returns a cloned individual to mimic mutation when no mutation logic is
    /// provided.
    ///
    /// # Examples
    /// ```
    /// use jeans::{Chromosome, Individual};
    /// let individual = Individual::from_parts(Chromosome::new(vec![0.0]), 0.0);
    /// let cloned = individual.mutate();
    /// assert_eq!(cloned.chromosome().genes(), individual.chromosome().genes());
    /// ```
    #[must_use]
    pub fn mutate(&self) -> Self {
        self.clone()
    }

    /// Performs crossover between two parents.
    ///
    /// # Examples
    /// ```
    /// use jeans::{Chromosome, Individual};
    /// use rand::thread_rng;
    /// let parent_a = Individual::from_parts(Chromosome::new(vec![0.0, 1.0]), 0.0);
    /// let parent_b = Individual::from_parts(Chromosome::new(vec![1.0, 0.0]), 0.0);
    /// let child = parent_a.cross(&parent_b, &mut thread_rng());
    /// assert_eq!(child.chromosome().len(), 2);
    /// ```
    #[must_use]
    pub fn cross(&self, other: &Self, rng: &mut impl Rng) -> Self {
        let mut genes = Vec::with_capacity(self.chromosome.len());
        for (gene_self, gene_other) in self.chromosome.iter().zip(other.chromosome.iter()) {
            if rng.gen::<f64>() < 0.5 {
                genes.push(*gene_self);
            } else {
                genes.push(*gene_other);
            }
        }
        let chromosome = Chromosome::new(genes);
        Self {
            chromosome,
            fitness: 0.0,
        }
    }
}
/// Collection of [`Individual`] values used by the optimizer.
///
/// # Examples
/// ```
/// use jeans::{Population, Settings};
/// let mut settings = Settings::default();
/// let population = Population::new(&mut settings);
/// assert_eq!(population.len() as u32, settings.population_size);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Population {
    individuals: Vec<Individual>,
}

impl Population {
    /// Creates a population populated with random individuals from the provided
    /// settings.
    ///
    /// # Examples
    /// ```
    /// use jeans::{Population, Settings};
    /// let mut settings = Settings::default();
    /// let population = Population::new(&mut settings);
    /// assert_eq!(population.len() as u32, settings.population_size);
    /// ```
    ///
    /// # Panics
    /// Panics when the provided settings contain inconsistent bounds that
    /// prevent creating individuals.
    #[must_use]
    pub fn new(settings: &mut crate::Settings) -> Self {
        Self::try_new(settings).expect("settings bounds are valid")
    }

    /// Attempts to create a population from the provided settings.
    ///
    /// # Examples
    /// ```
    /// use jeans::{Population, Settings};
    /// let mut settings = Settings::default();
    /// let population = Population::try_new(&mut settings).unwrap();
    /// assert_eq!(population.len() as u32, settings.population_size);
    /// ```
    ///
    /// # Errors
    /// Returns [`BoundsError`] when the settings contain inconsistent bounds.
    ///
    /// # Panics
    /// Panics when the configured population size exceeds what can be
    /// represented on the current platform.
    pub fn try_new(settings: &mut crate::Settings) -> Result<Self, BoundsError> {
        let capacity =
            usize::try_from(settings.population_size).expect("population size must fit into usize");
        let mut individuals = Vec::with_capacity(capacity);
        for _ in 0..settings.population_size {
            individuals.push(Individual::try_new(settings)?);
        }
        Ok(Self { individuals })
    }

    /// Creates an empty population.
    ///
    /// # Examples
    /// ```
    /// use jeans::Population;
    /// let population = Population::empty();
    /// assert_eq!(population.len(), 0);
    /// ```
    #[must_use]
    pub fn empty() -> Self {
        Self {
            individuals: Vec::new(),
        }
    }

    /// Returns the number of individuals in the population.
    ///
    /// # Examples
    /// ```
    /// use jeans::{Population, Settings};
    /// let mut settings = Settings::default();
    /// let population = Population::new(&mut settings);
    /// assert_eq!(population.len() as u32, settings.population_size);
    /// ```
    #[must_use]
    pub fn len(&self) -> usize {
        self.individuals.len()
    }

    /// Indicates whether the population is empty.
    ///
    /// # Examples
    /// ```
    /// use jeans::Population;
    /// let population = Population::empty();
    /// assert!(population.is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.individuals.is_empty()
    }

    /// Adds an individual to the population.
    ///
    /// # Examples
    /// ```
    /// use jeans::{Chromosome, Individual, Population};
    /// let mut population = Population::empty();
    /// population.push(Individual::from_parts(Chromosome::new(vec![0.0]), 0.0));
    /// assert_eq!(population.len(), 1);
    /// ```
    pub fn push(&mut self, individual: Individual) {
        self.individuals.push(individual);
    }

    /// Returns a shallow copy of the population.
    ///
    /// # Examples
    /// ```
    /// use jeans::{Population, Settings};
    /// let mut settings = Settings::default();
    /// let population = Population::new(&mut settings);
    /// let clone = population.copy();
    /// assert_eq!(clone.len(), population.len());
    /// ```
    #[must_use]
    pub fn copy(&self) -> Self {
        Self {
            individuals: self.individuals.clone(),
        }
    }

    /// Returns the best individual according to fitness ordering.
    ///
    /// # Examples
    /// ```
    /// use jeans::{Population, Settings};
    /// let mut settings = Settings::default();
    /// let population = Population::new(&mut settings);
    /// assert!(population.best_individual().is_some());
    /// ```
    #[must_use]
    pub fn best_individual(&self) -> Option<&Individual> {
        self.individuals.iter().max_by(|lhs, rhs| {
            lhs.fitness()
                .partial_cmp(&rhs.fitness())
                .unwrap_or(Ordering::Equal)
        })
    }

    /// Returns the best fitness found so far.
    ///
    /// # Examples
    /// ```
    /// use jeans::{Population, Settings};
    /// let mut settings = Settings::default();
    /// let population = Population::new(&mut settings);
    /// assert!(population.get_best().is_finite());
    /// ```
    #[must_use]
    pub fn get_best(&self) -> Gene {
        self.best_individual()
            .map_or(-f64::INFINITY, Individual::fitness)
    }

    /// Returns the mean fitness of the population.
    ///
    /// # Examples
    /// ```
    /// use jeans::{Population, Settings};
    /// let mut settings = Settings::default();
    /// let population = Population::new(&mut settings);
    /// assert!(population.get_mean().is_finite());
    /// ```
    #[must_use]
    pub fn get_mean(&self) -> Gene {
        if self.individuals.is_empty() {
            return 0.0;
        }
        let sum: Gene = self.individuals.iter().map(Individual::fitness).sum();
        sum / Self::len_as_gene(self.individuals.len())
    }

    /// Returns the population standard deviation of the fitness values.
    ///
    /// # Examples
    /// ```
    /// use jeans::{Population, Settings};
    /// let mut settings = Settings::default();
    /// let population = Population::new(&mut settings);
    /// assert!(population.get_std().is_finite());
    /// ```
    #[must_use]
    pub fn get_std(&self) -> Gene {
        if self.individuals.len() <= 1 {
            return 0.0;
        }
        let mean = self.get_mean();
        let variance: Gene = self
            .individuals
            .iter()
            .map(|individual| {
                let diff = individual.fitness() - mean;
                diff * diff
            })
            .sum::<Gene>()
            / Self::len_as_gene(self.individuals.len());
        variance.sqrt()
    }

    /// Returns a random individual.
    ///
    /// # Examples
    /// ```
    /// use jeans::{Population, Settings};
    /// let mut settings = Settings::default();
    /// let population = Population::new(&mut settings);
    /// let _ = population.get_random();
    /// ```
    ///
    /// # Panics
    /// Panics when called on an empty population.
    #[must_use]
    pub fn get_random(&self) -> Individual {
        let mut rng = rand::thread_rng();
        self.individuals
            .choose(&mut rng)
            .expect("cannot sample from an empty population")
            .clone()
    }

    /// Sorts the population by fitness.
    ///
    /// # Examples
    /// ```
    /// use jeans::{Population, Settings};
    /// let mut settings = Settings::default();
    /// let mut population = Population::new(&mut settings);
    /// population.sort();
    /// ```
    ///
    /// # Panics
    /// Panics if any fitness value is `NaN`, because the comparison cannot be
    /// completed.
    pub fn sort(&mut self) {
        self.individuals
            .sort_unstable_by(|lhs, rhs| lhs.fitness().partial_cmp(&rhs.fitness()).unwrap());
    }

    /// Returns the underlying individuals.
    ///
    /// # Examples
    /// ```
    /// use jeans::{Population, Settings};
    /// let mut settings = Settings::default();
    /// let population = Population::new(&mut settings);
    /// assert_eq!(population.individuals().len(), population.len());
    /// ```
    #[must_use]
    pub fn individuals(&self) -> &[Individual] {
        &self.individuals
    }

    fn len_as_gene(len: usize) -> Gene {
        #[allow(clippy::cast_precision_loss)]
        {
            len as Gene
        }
    }
}
/// Error returned when invalid bounds are provided.
///
/// # Examples
/// ```
/// use jeans::Chromosome;
/// let err = Chromosome::random_with_bounds(&[0.0, 0.0], &[1.0], &mut rand::thread_rng()).unwrap_err();
/// assert!(err.to_string().contains("dimension mismatch"));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum BoundsError {
    /// The number of provided bounds entries does not match the expected
    /// dimensionality.
    DimensionMismatch {
        /// Number of dimensions specified by the caller.
        expected: usize,
        /// Number of bounds entries actually provided.
        found: usize,
    },
    /// One of the dimensions has an invalid lower/upper pairing.
    InvalidRange {
        /// The index of the problematic dimension.
        dimension: usize,
        /// The invalid lower bound value.
        lower: Gene,
        /// The invalid upper bound value.
        upper: Gene,
    },
}

impl std::fmt::Display for BoundsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BoundsError::DimensionMismatch { expected, found } => write!(
                f,
                "dimension mismatch: expected {expected} bounds entries but found {found}"
            ),
            BoundsError::InvalidRange {
                dimension,
                lower,
                upper,
            } => write!(
                f,
                "invalid bounds for dimension {dimension} (lower: {lower}, upper: {upper})"
            ),
        }
    }
}

impl std::error::Error for BoundsError {}

fn validate_bounds(lower_bounds: &[Gene], upper_bounds: &[Gene]) -> Result<(), BoundsError> {
    if lower_bounds.len() != upper_bounds.len() {
        return Err(BoundsError::DimensionMismatch {
            expected: lower_bounds.len(),
            found: upper_bounds.len(),
        });
    }
    for (idx, (lower, upper)) in lower_bounds.iter().zip(upper_bounds.iter()).enumerate() {
        if lower > upper {
            return Err(BoundsError::InvalidRange {
                dimension: idx,
                lower: *lower,
                upper: *upper,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chromosome_random_respects_bounds() {
        let lower = vec![0.0, -1.0, 10.0];
        let upper = vec![1.0, 1.0, 10.0];
        let chromosome =
            Chromosome::random_with_bounds(&lower, &upper, &mut rand::thread_rng()).unwrap();
        assert_eq!(chromosome.len(), 3);
        for ((gene, lower), upper) in chromosome.iter().zip(lower.iter()).zip(upper.iter()) {
            assert!(*lower <= *gene && *gene <= *upper);
        }
    }

    #[test]
    fn population_statistics_behave_reasonably() {
        let mut individuals = vec![];
        for idx in 0..5 {
            let chromosome = Chromosome::new(vec![idx as Gene]);
            individuals.push(Individual::from_parts(chromosome, idx as Gene));
        }
        let population = Population { individuals };
        assert_eq!(population.get_best(), 4.0);
        assert_eq!(population.get_mean(), 2.0);
        assert!(population.get_std() > 0.0);
        assert!(population.best_individual().is_some());
    }

    #[test]
    fn validate_bounds_errors() {
        let err = Chromosome::random_with_bounds(&[0.0], &[], &mut rand::thread_rng()).unwrap_err();
        assert!(matches!(err, BoundsError::DimensionMismatch { .. }));
    }
}
