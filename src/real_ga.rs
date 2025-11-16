//! High-level real-coded genetic algorithm engine.
//!
//! The [`RealGa`] builder exposes a strongly-typed interface that wires the
//! modular operators defined in [`crate::ops`] together.  Users construct the
//! engine through [`RealGa::builder`], customize the operators or stop
//! conditions, and then call [`RealGa::run`] with a random number generator to
//! perform the optimization.

use crate::core::{
    population_diversity_by, ExperimentMetadata, ExperimentResult, IndividualSnapshot, RunStats,
};
use crate::ops::{
    CrossoverOperator, MutationOperator, OperatorError, PolynomialMutation, ProblemError,
    SelectionOperator, SimulatedBinaryCrossover,
};
use crate::r#async::{EvaluationError, SingleObjectiveEvaluator};
use rand::distributions::Uniform;
use rand::{Rng, RngCore};
use std::fmt::{self, Display, Formatter};

const DEFAULT_SBX_ETA: f64 = 15.0;
const DEFAULT_POLY_ETA: f64 = 20.0;
const DEFAULT_TOURNAMENT_SIZE: usize = 3;
const DEFAULT_GENERATIONS: usize = 100;

/// Reports produced after running [`RealGa::run`].
///
/// # Examples
/// ```
/// use jeans::{RealGa, StopCondition};
/// use jeans::ops::{Problem, ProblemBounds, ProblemResult};
/// use rand::SeedableRng;
///
/// struct Quadratic;
///
/// impl ProblemBounds for Quadratic {
///     fn dimensions(&self) -> usize { 1 }
///     fn lower_bounds(&self) -> &[f64] { &[-5.0] }
///     fn upper_bounds(&self) -> &[f64] { &[5.0] }
/// }
///
/// impl Problem for Quadratic {
///     fn evaluate(&mut self, genes: &[f64]) -> ProblemResult<f64> {
///         Ok(genes[0] * genes[0])
///     }
/// }
///
/// let problem = Quadratic;
/// let mut ga = RealGa::builder(problem)
///     .stop_condition(StopCondition::max_generations(2))
///     .build()
///     .unwrap();
/// let mut rng = rand::rngs::StdRng::seed_from_u64(7);
/// let report = ga.run(&mut rng).unwrap();
/// assert!(report.best_fitness >= 0.0);
/// assert_eq!(report.generations, report.experiment.metadata.generations);
/// assert!(!report.experiment.stats.best_fitness.is_empty());
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct RealGaReport {
    /// Best chromosome discovered by the engine.
    pub best_solution: Vec<f64>,
    /// Fitness associated with [`Self::best_solution`].
    pub best_fitness: f64,
    /// Number of generations executed before stopping.
    pub generations: usize,
    /// Complete experimentation payload for downstream analysis.
    pub experiment: ExperimentResult<f64>,
}

/// Errors produced by the [`RealGa`] engine or its default operators.
#[derive(Debug)]
pub enum RealGaError {
    /// The configured population size was zero.
    InvalidPopulationSize(usize),
    /// Distribution index used by either SBX or polynomial mutation was invalid.
    InvalidDistributionIndex {
        /// Name of the operator reporting the error.
        operator: &'static str,
        /// Offending value.
        value: f64,
    },
    /// Mutation probability was outside `[0, 1]`.
    InvalidMutationProbability(f64),
    /// Tournament selection received an empty tournament size.
    InvalidTournamentSize(usize),
    /// Operator parameter failed validation.
    InvalidOperatorParameter {
        /// Name of the operator reporting the error.
        operator: &'static str,
        /// Name of the invalid parameter.
        parameter: &'static str,
        /// Offending value.
        value: f64,
    },
    /// Selection operator failed to return parents for reproduction.
    SelectionFailed,
    /// Wrapper around [`crate::core::BoundsError`].
    Bounds(crate::core::BoundsError),
    /// Wrapper around [`ProblemError`].
    Problem(ProblemError),
    /// Wrapper around asynchronous evaluation errors.
    Evaluation(EvaluationError),
}

impl Display for RealGaError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPopulationSize(size) => {
                write!(
                    f,
                    "population size must be greater than zero (received {size})"
                )
            }
            Self::InvalidDistributionIndex { operator, value } => {
                write!(
                    f,
                    "{operator} distribution index must be positive (received {value})"
                )
            }
            Self::InvalidMutationProbability(prob) => {
                write!(
                    f,
                    "mutation probability must be within [0, 1] (received {prob})"
                )
            }
            Self::InvalidOperatorParameter {
                operator,
                parameter,
                value,
            } => {
                write!(
                    f,
                    "{operator} parameter {parameter} was invalid (received {value})"
                )
            }
            Self::InvalidTournamentSize(size) => {
                write!(f, "tournament size must be at least one (received {size})")
            }
            Self::SelectionFailed => f.write_str("selection operator failed to provide parents"),
            Self::Bounds(err) => write!(f, "{err}"),
            Self::Problem(err) => write!(f, "{err}"),
            Self::Evaluation(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for RealGaError {}

impl From<crate::core::BoundsError> for RealGaError {
    fn from(err: crate::core::BoundsError) -> Self {
        Self::Bounds(err)
    }
}

impl From<ProblemError> for RealGaError {
    fn from(err: ProblemError) -> Self {
        Self::Problem(err)
    }
}

impl From<EvaluationError> for RealGaError {
    fn from(err: EvaluationError) -> Self {
        match err {
            EvaluationError::Problem(problem) => Self::Problem(problem),
            other => Self::Evaluation(other),
        }
    }
}

impl From<OperatorError> for RealGaError {
    fn from(err: OperatorError) -> Self {
        match err {
            OperatorError::InvalidDistributionIndex { operator, value } => {
                Self::InvalidDistributionIndex { operator, value }
            }
            OperatorError::InvalidProbability { operator, value } => {
                if operator.contains("mutation") {
                    Self::InvalidMutationProbability(value)
                } else {
                    Self::InvalidOperatorParameter {
                        operator,
                        parameter: "probability",
                        value,
                    }
                }
            }
            OperatorError::InvalidParameter {
                operator,
                parameter,
                value,
            } => Self::InvalidOperatorParameter {
                operator,
                parameter,
                value,
            },
            OperatorError::Bounds(err) => Self::Bounds(err),
        }
    }
}

/// Combines primitive stop conditions.
#[derive(Debug, Clone)]
pub enum StopCondition {
    /// Stop when the generation counter reaches this limit.
    MaxGenerations {
        /// Maximum number of generations to run.
        limit: usize,
    },
    /// Stop when the best fitness drops below the provided value.
    TargetFitnessBelow {
        /// Target fitness that ends the run when best fitness is below or equal.
        threshold: f64,
    },
    /// Logical OR that triggers when either child condition is met.
    Or(Box<StopCondition>, Box<StopCondition>),
}

impl StopCondition {
    /// Creates a stop condition that limits the maximum number of generations.
    #[must_use]
    pub fn max_generations(limit: usize) -> Self {
        Self::MaxGenerations { limit }
    }

    /// Creates a stop condition that targets a best fitness threshold.
    #[must_use]
    pub fn target_fitness_below(threshold: f64) -> Self {
        Self::TargetFitnessBelow { threshold }
    }

    /// Combines two stop conditions using logical OR semantics.
    #[must_use]
    pub fn or(self, other: StopCondition) -> Self {
        Self::Or(Box::new(self), Box::new(other))
    }

    fn is_met(&self, generations: usize, best_fitness: f64) -> bool {
        match self {
            Self::MaxGenerations { limit } => generations >= *limit,
            Self::TargetFitnessBelow { threshold } => best_fitness <= *threshold,
            Self::Or(left, right) => {
                left.is_met(generations, best_fitness) || right.is_met(generations, best_fitness)
            }
        }
    }
}

/// Builder returned by [`RealGa::builder`].
pub struct RealGaBuilder<P> {
    problem: P,
    population_size: usize,
    crossover: Option<Box<dyn CrossoverOperator>>,
    mutation: Option<Box<dyn MutationOperator>>,
    selection: Option<Box<dyn SelectionOperator>>,
    stop_condition: StopCondition,
    lower_bounds: Vec<f64>,
    upper_bounds: Vec<f64>,
}

impl<P> RealGaBuilder<P>
where
    P: SingleObjectiveEvaluator,
{
    /// Configures the number of individuals per generation.
    #[must_use]
    pub fn population_size(mut self, size: usize) -> Self {
        self.population_size = size;
        self
    }

    /// Replaces the crossover operator.
    #[must_use]
    pub fn crossover(mut self, operator: impl CrossoverOperator + 'static) -> Self {
        self.crossover = Some(Box::new(operator));
        self
    }

    /// Replaces the mutation operator.
    #[must_use]
    pub fn mutation(mut self, operator: impl MutationOperator + 'static) -> Self {
        self.mutation = Some(Box::new(operator));
        self
    }

    /// Replaces the selection operator.
    #[must_use]
    pub fn selection(mut self, operator: impl SelectionOperator + 'static) -> Self {
        self.selection = Some(Box::new(operator));
        self
    }

    /// Configures the stop condition evaluated during [`RealGa::run`].
    #[must_use]
    pub fn stop_condition(mut self, condition: StopCondition) -> Self {
        self.stop_condition = condition;
        self
    }

    /// Finalizes the builder into a [`RealGa`] engine.
    ///
    /// # Errors
    /// Returns [`RealGaError`] when the population size is zero, the problem
    /// bounds are inconsistent, or default operator construction fails.
    pub fn build(self) -> Result<RealGa<P>, RealGaError> {
        if self.population_size == 0 {
            return Err(RealGaError::InvalidPopulationSize(0));
        }
        if self.lower_bounds.len() != self.upper_bounds.len() {
            return Err(RealGaError::Bounds(
                crate::core::BoundsError::DimensionMismatch {
                    expected: self.lower_bounds.len(),
                    found: self.upper_bounds.len(),
                },
            ));
        }
        let problem = self.problem;
        problem.validate_bounds()?;
        let dims = problem.dimensions();
        let crossover: Box<dyn CrossoverOperator> = match self.crossover {
            Some(operator) => operator,
            None => Box::new(SimulatedBinaryCrossover::new(DEFAULT_SBX_ETA)?),
        };
        let mutation = if let Some(operator) = self.mutation {
            operator
        } else {
            #[allow(clippy::cast_precision_loss)]
            let probability = if dims == 0 { 1.0 } else { 1.0 / dims as f64 };
            Box::new(PolynomialMutation::new(
                self.lower_bounds.clone(),
                self.upper_bounds.clone(),
                DEFAULT_POLY_ETA,
                probability,
            )?)
        };
        let selection: Box<dyn SelectionOperator> = match self.selection {
            Some(operator) => operator,
            None => Box::new(TournamentSelection::new(DEFAULT_TOURNAMENT_SIZE)?),
        };
        Ok(RealGa {
            problem,
            population_size: self.population_size,
            crossover,
            mutation,
            selection,
            stop_condition: self.stop_condition,
            lower_bounds: self.lower_bounds,
            upper_bounds: self.upper_bounds,
        })
    }
}

/// Real-coded genetic algorithm engine that operates over a [`SingleObjectiveEvaluator`].
pub struct RealGa<P> {
    problem: P,
    population_size: usize,
    crossover: Box<dyn CrossoverOperator>,
    mutation: Box<dyn MutationOperator>,
    selection: Box<dyn SelectionOperator>,
    stop_condition: StopCondition,
    lower_bounds: Vec<f64>,
    upper_bounds: Vec<f64>,
}

impl<P> RealGa<P>
where
    P: SingleObjectiveEvaluator,
{
    /// Creates a builder used to configure the engine.
    #[must_use]
    pub fn builder(problem: P) -> RealGaBuilder<P> {
        let lower_bounds = problem.lower_bounds().to_vec();
        let upper_bounds = problem.upper_bounds().to_vec();
        RealGaBuilder {
            problem,
            population_size: 50,
            crossover: None,
            mutation: None,
            selection: None,
            stop_condition: StopCondition::max_generations(DEFAULT_GENERATIONS),
            lower_bounds,
            upper_bounds,
        }
    }

    /// Runs the genetic algorithm using the provided random number generator.
    ///
    /// # Errors
    /// Propagates any [`RealGaError`] emitted by problem evaluation or the
    /// configured operators.
    pub fn run<R: Rng>(&mut self, rng: &mut R) -> Result<RealGaReport, RealGaError> {
        let mut population = self.initialize_population(rng);
        let mut fitness = self.evaluate_population(&population)?;
        let mut best_solution = Vec::new();
        let mut best_fitness = f64::INFINITY;
        let mut has_best =
            Self::update_best(&population, &fitness, &mut best_solution, &mut best_fitness);
        let mut stats = RunStats::new();
        record_scalar_stats(&mut stats, &population, &fitness);
        let mut generation = 0_usize;
        while !self.stop_condition.is_met(generation, best_fitness) {
            generation = generation.saturating_add(1);
            let mut offspring = Vec::with_capacity(self.population_size);
            let selection_scores = Self::selection_scores(&fitness);
            while offspring.len() < self.population_size {
                let pair = self
                    .selection
                    .select_pair(&selection_scores, rng)
                    .ok_or(RealGaError::SelectionFailed)?;
                let parent_a = &population[pair.0];
                let parent_b = &population[pair.1];
                let (child_a, child_b) = self.crossover.crossover(parent_a, parent_b, rng);
                let mut child_a = self.mutation.mutate(&child_a, rng);
                self.clamp(&mut child_a);
                offspring.push(child_a);
                if offspring.len() >= self.population_size {
                    break;
                }
                let mut child_b = self.mutation.mutate(&child_b, rng);
                self.clamp(&mut child_b);
                offspring.push(child_b);
            }
            population = offspring;
            fitness = self.evaluate_population(&population)?;
            has_best |=
                Self::update_best(&population, &fitness, &mut best_solution, &mut best_fitness);
            record_scalar_stats(&mut stats, &population, &fitness);
        }
        let best_individual =
            has_best.then(|| IndividualSnapshot::new(best_solution.clone(), best_fitness));
        let final_population = population
            .iter()
            .zip(fitness.iter().copied())
            .map(|(genes, fitness)| IndividualSnapshot::new(genes.clone(), fitness))
            .collect();
        let metadata = ExperimentMetadata::new(generation, None, std::any::type_name::<R>());
        let experiment = ExperimentResult {
            final_population,
            best_individual,
            stats,
            metadata,
        };
        Ok(RealGaReport {
            best_solution,
            best_fitness,
            generations: generation,
            experiment,
        })
    }

    fn initialize_population<R: Rng>(&self, rng: &mut R) -> Vec<Vec<f64>> {
        let mut population = Vec::with_capacity(self.population_size);
        for _ in 0..self.population_size {
            population.push(self.random_candidate(rng));
        }
        population
    }

    fn random_candidate<R: Rng>(&self, rng: &mut R) -> Vec<f64> {
        let mut genes = Vec::with_capacity(self.lower_bounds.len());
        for (&lower, &upper) in self.lower_bounds.iter().zip(&self.upper_bounds) {
            let sampler = Uniform::new_inclusive(lower, upper);
            genes.push(rng.sample(sampler));
        }
        genes
    }

    fn clamp(&self, candidate: &mut [f64]) {
        for ((value, &lower), &upper) in candidate
            .iter_mut()
            .zip(self.lower_bounds.iter())
            .zip(self.upper_bounds.iter())
        {
            *value = value.clamp(lower, upper);
        }
    }

    fn evaluate_population(&mut self, population: &[Vec<f64>]) -> Result<Vec<f64>, RealGaError> {
        Ok(self.problem.evaluate_population(population)?)
    }

    fn update_best(
        population: &[Vec<f64>],
        fitness: &[f64],
        best_solution: &mut Vec<f64>,
        best_fitness: &mut f64,
    ) -> bool {
        let mut improved = false;
        if let Some((idx, &value)) = fitness
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
        {
            if value < *best_fitness {
                *best_fitness = value;
                best_solution.clone_from(&population[idx]);
                improved = true;
            }
        }
        improved
    }

    fn selection_scores(fitness: &[f64]) -> Vec<f64> {
        fitness.iter().map(|value| -*value).collect()
    }
}

fn record_scalar_stats(stats: &mut RunStats<f64>, population: &[Vec<f64>], fitness: &[f64]) {
    stats
        .population_diversity
        .push(population_diversity_by(population.len(), |idx| {
            population[idx].as_slice()
        }));
    if fitness.is_empty() {
        stats.best_fitness.push(f64::NAN);
        stats.mean_fitness.push(f64::NAN);
        stats.median_fitness.push(f64::NAN);
        return;
    }
    stats.best_fitness.push(best_fitness_value(fitness));
    stats.mean_fitness.push(mean_fitness_value(fitness));
    stats.median_fitness.push(median_fitness_value(fitness));
}

fn best_fitness_value(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::INFINITY, f64::min)
}

fn mean_fitness_value(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    #[allow(clippy::cast_precision_loss)]
    {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn median_fitness_value(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        f64::midpoint(sorted[mid - 1], sorted[mid])
    } else {
        sorted[mid]
    }
}

/// Tournament selection that maximizes the provided fitness scores.
#[derive(Debug, Clone)]
pub struct TournamentSelection {
    size: usize,
}

impl TournamentSelection {
    /// Creates a tournament selector of the provided size.
    ///
    /// # Errors
    /// Returns [`RealGaError::InvalidTournamentSize`] when `size` is zero.
    pub fn new(size: usize) -> Result<Self, RealGaError> {
        if size == 0 {
            return Err(RealGaError::InvalidTournamentSize(size));
        }
        Ok(Self { size })
    }
}

impl SelectionOperator for TournamentSelection {
    fn select_index(&self, fitness_values: &[f64], rng: &mut dyn RngCore) -> Option<usize> {
        if fitness_values.is_empty() {
            return None;
        }
        let tournament = self.size.min(fitness_values.len());
        let mut best_idx = None;
        for _ in 0..tournament {
            let idx = random_index(fitness_values.len(), rng);
            best_idx = match best_idx {
                Some(current) => {
                    if fitness_values[idx] > fitness_values[current] {
                        Some(idx)
                    } else {
                        Some(current)
                    }
                }
                None => Some(idx),
            };
        }
        best_idx
    }
}

fn random_index(len: usize, rng: &mut dyn RngCore) -> usize {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    {
        (rng.next_u64() as usize) % len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{AsyncProblem, Problem, ProblemBounds};
    use crate::r#async::AsyncBatchEvaluator;
    use async_trait::async_trait;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::time::Duration;

    struct Sphere {
        lower: Vec<f64>,
        upper: Vec<f64>,
    }

    impl Sphere {
        fn new(dimensions: usize) -> Self {
            Self {
                lower: vec![-5.0; dimensions],
                upper: vec![5.0; dimensions],
            }
        }
    }

    impl ProblemBounds for Sphere {
        fn dimensions(&self) -> usize {
            self.lower.len()
        }

        fn lower_bounds(&self) -> &[f64] {
            &self.lower
        }

        fn upper_bounds(&self) -> &[f64] {
            &self.upper
        }
    }

    impl Problem for Sphere {
        fn evaluate(&mut self, genes: &[f64]) -> crate::ops::ProblemResult<f64> {
            Ok(genes.iter().map(|value| value * value).sum())
        }
    }

    #[test]
    fn stop_condition_max_generations() {
        let condition = StopCondition::max_generations(2);
        assert!(!condition.is_met(0, 0.0));
        assert!(condition.is_met(2, 10.0));
    }

    #[test]
    fn stop_condition_target_fitness() {
        let condition = StopCondition::target_fitness_below(0.1);
        assert!(condition.is_met(0, 0.05));
        assert!(!condition.is_met(0, 0.5));
    }

    #[test]
    fn stop_condition_or_logic() {
        let condition =
            StopCondition::max_generations(1).or(StopCondition::target_fitness_below(1.0));
        assert!(condition.is_met(1, 5.0));
        assert!(condition.is_met(0, 0.5));
    }

    #[test]
    fn real_ga_respects_generation_limit() {
        let problem = Sphere::new(2);
        let mut ga = RealGa::builder(problem)
            .population_size(6)
            .stop_condition(StopCondition::max_generations(2))
            .build()
            .unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        let report = ga.run(&mut rng).unwrap();
        assert!(report.generations <= 2);
        assert_eq!(report.best_solution.len(), 2);
        assert!(report.best_fitness.is_finite());
        assert_eq!(report.generations, report.experiment.metadata.generations);
        assert_eq!(report.experiment.final_population.len(), 6);
        assert_eq!(
            report.experiment.stats.generations(),
            report.experiment.stats.best_fitness.len()
        );
        assert!(report.experiment.best_individual.is_some());
    }

    #[test]
    fn target_fitness_stop_triggers_immediately() {
        struct Zero;

        impl ProblemBounds for Zero {
            fn dimensions(&self) -> usize {
                1
            }
            fn lower_bounds(&self) -> &[f64] {
                &[0.0]
            }
            fn upper_bounds(&self) -> &[f64] {
                &[0.0]
            }
        }

        impl Problem for Zero {
            fn evaluate(&mut self, _genes: &[f64]) -> crate::ops::ProblemResult<f64> {
                Ok(0.0)
            }
        }

        let problem = Zero;
        let mut ga = RealGa::builder(problem)
            .population_size(2)
            .stop_condition(StopCondition::target_fitness_below(0.1))
            .build()
            .unwrap();
        let mut rng = StdRng::seed_from_u64(7);
        let report = ga.run(&mut rng).unwrap();
        assert_eq!(report.generations, 0);
        assert!(report.best_fitness <= 0.1);
        assert_eq!(report.experiment.stats.best_fitness.len(), 1);
        assert!(report.experiment.stats.population_diversity[0] <= 0.0);
    }

    #[test]
    fn real_ga_handles_async_problems() {
        struct AsyncSphere;

        impl ProblemBounds for AsyncSphere {
            fn dimensions(&self) -> usize {
                2
            }

            fn lower_bounds(&self) -> &[f64] {
                &[-5.0, -5.0]
            }

            fn upper_bounds(&self) -> &[f64] {
                &[5.0, 5.0]
            }
        }

        #[async_trait]
        impl AsyncProblem for AsyncSphere {
            async fn evaluate_async(&self, genes: &[f64]) -> crate::ops::ProblemResult<f64> {
                tokio::time::sleep(Duration::from_millis(5)).await;
                Ok(genes.iter().map(|value| value * value).sum())
            }
        }

        let evaluator = AsyncBatchEvaluator::with_max_concurrency(AsyncSphere, 2).unwrap();
        let mut ga = RealGa::builder(evaluator)
            .population_size(4)
            .stop_condition(StopCondition::max_generations(1))
            .build()
            .unwrap();
        let mut rng = StdRng::seed_from_u64(5);
        let report = ga.run(&mut rng).unwrap();
        assert_eq!(report.generations, 1);
        assert_eq!(report.best_solution.len(), 2);
        assert!(report.best_fitness.is_finite());
    }

    #[test]
    fn median_fitness_value_handles_even_population_sizes() {
        let fitness = [4.0, 8.0, 1.0, 7.0];
        let median = median_fitness_value(&fitness);
        assert!((median - 5.5).abs() <= f64::EPSILON);
    }

    #[test]
    fn median_fitness_value_handles_odd_population_sizes() {
        let fitness = [3.0, 9.0, 1.0];
        let median = median_fitness_value(&fitness);
        assert!((median - 3.0).abs() <= f64::EPSILON);
    }

    #[test]
    fn median_fitness_value_returns_nan_for_empty_populations() {
        let median = median_fitness_value(&[]);
        assert!(median.is_nan());
    }
}
