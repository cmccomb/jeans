//! Shared experimentation metadata and analytics utilities.
//!
//! These types capture time-series metrics emitted by optimization engines and
//! provide a consistent payload for downstream analysis. They are
//! intentionally lightweight to keep serialization overhead minimal.

/// Time-series metrics captured during an optimization run.
///
/// # Examples
/// ```
/// use jeans::RunStats;
/// let stats: RunStats<f64> = RunStats::new();
/// assert_eq!(stats.generations(), 0);
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct RunStats<T> {
    /// Best fitness value observed in each generation.
    pub best_fitness: Vec<T>,
    /// Mean fitness value computed from the full population per generation.
    pub mean_fitness: Vec<T>,
    /// Median fitness value computed per generation.
    pub median_fitness: Vec<T>,
    /// Diversity score of the population per generation.
    pub population_diversity: Vec<f64>,
}

impl<T> RunStats<T> {
    /// Creates an empty set of run statistics.
    #[must_use]
    pub fn new() -> Self {
        Self {
            best_fitness: Vec::new(),
            mean_fitness: Vec::new(),
            median_fitness: Vec::new(),
            population_diversity: Vec::new(),
        }
    }

    /// Returns the number of generations tracked by the stats object.
    #[must_use]
    pub fn generations(&self) -> usize {
        self.best_fitness.len()
    }
}

impl<T> Default for RunStats<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Metadata describing an executed experiment.
///
/// # Examples
/// ```
/// use jeans::ExperimentMetadata;
/// let metadata = ExperimentMetadata::new(10, Some(7), "StdRng");
/// assert_eq!(metadata.generations, 10);
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct ExperimentMetadata {
    /// Number of generations executed by the experiment.
    pub generations: usize,
    /// Optional RNG seed recorded by the caller.
    pub rng_seed: Option<u64>,
    /// Human readable description of the RNG used for the run.
    pub rng_description: String,
}

impl ExperimentMetadata {
    /// Creates a new metadata record.
    #[must_use]
    pub fn new(
        generations: usize,
        rng_seed: Option<u64>,
        rng_description: impl Into<String>,
    ) -> Self {
        Self {
            generations,
            rng_seed,
            rng_description: rng_description.into(),
        }
    }
}

/// Minimal representation of an individual with its associated fitness.
///
/// # Examples
/// ```
/// use jeans::IndividualSnapshot;
/// let snapshot = IndividualSnapshot::new(vec![0.0, 1.0], 0.5);
/// assert_eq!(snapshot.genes.len(), 2);
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct IndividualSnapshot<F> {
    /// Genes that represent the individual.
    pub genes: Vec<f64>,
    /// Fitness or objectives associated with the genes.
    pub fitness: F,
}

impl<F> IndividualSnapshot<F> {
    /// Creates an [`IndividualSnapshot`] from genes and a fitness payload.
    #[must_use]
    pub fn new(genes: Vec<f64>, fitness: F) -> Self {
        Self { genes, fitness }
    }
}

/// Complete experiment payload returned by the engines.
///
/// # Examples
/// ```
/// use jeans::{ExperimentMetadata, ExperimentResult, IndividualSnapshot, RunStats};
/// let result = ExperimentResult {
///     final_population: vec![IndividualSnapshot::new(vec![0.0], 0.0)],
///     best_individual: None,
///     stats: RunStats::new(),
///     metadata: ExperimentMetadata::new(1, None, "rng"),
/// };
/// assert_eq!(result.final_population.len(), 1);
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct ExperimentResult<F> {
    /// Final population at the end of the experiment.
    pub final_population: Vec<IndividualSnapshot<F>>,
    /// Best individual discovered during the run, when applicable.
    pub best_individual: Option<IndividualSnapshot<F>>,
    /// Time-series metrics recorded during the run.
    pub stats: RunStats<F>,
    /// Metadata captured from the execution environment.
    pub metadata: ExperimentMetadata,
}

pub(crate) fn population_diversity_by<'a, F>(size: usize, mut at: F) -> f64
where
    F: FnMut(usize) -> &'a [f64],
{
    if size == 0 {
        return 0.0;
    }
    let dimensions = at(0).len();
    if dimensions == 0 {
        return 0.0;
    }
    let mut means = vec![0.0; dimensions];
    #[allow(clippy::cast_precision_loss)]
    let population_size = size as f64;
    for idx in 0..size {
        for (dimension, value) in at(idx).iter().enumerate() {
            means[dimension] += *value;
        }
    }
    for mean in &mut means {
        *mean /= population_size;
    }
    let mut total_variance = 0.0;
    for idx in 0..size {
        for (dimension, value) in at(idx).iter().enumerate() {
            let diff = value - means[dimension];
            total_variance += (diff * diff) / population_size;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    {
        (total_variance / dimensions as f64).sqrt()
    }
}
