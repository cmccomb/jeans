//! High-level real-coded genetic algorithm engine.
//!
//! The [`RealGa`] builder exposes a strongly-typed interface that wires the
//! modular operators defined in [`crate::ops`] together.  Users construct the
//! engine through [`RealGa::builder`], customize the operators or stop
//! conditions, and then call [`RealGa::run`] with a random number generator to
//! perform the optimization.

use crate::ops::{CrossoverOperator, MutationOperator, Problem, ProblemError, SelectionOperator};
use rand::distributions::Uniform;
use rand::{Rng, RngCore};
use std::fmt::{self, Display, Formatter};

const DEFAULT_SBX_ETA: f64 = 15.0;
const DEFAULT_POLY_ETA: f64 = 20.0;
const DEFAULT_TOURNAMENT_SIZE: usize = 3;
const DEFAULT_GENERATIONS: usize = 100;
#[allow(clippy::cast_precision_loss)]
const RNG_SCALE: f64 = 1.0 / (u64::MAX as f64 + 1.0);

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
/// ```
#[derive(Debug, Clone)]
pub struct RealGaReport {
    /// Best chromosome discovered by the engine.
    pub best_solution: Vec<f64>,
    /// Fitness associated with [`Self::best_solution`].
    pub best_fitness: f64,
    /// Number of generations executed before stopping.
    pub generations: usize,
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
    /// Selection operator failed to return parents for reproduction.
    SelectionFailed,
    /// Wrapper around [`crate::core::BoundsError`].
    Bounds(crate::core::BoundsError),
    /// Wrapper around [`ProblemError`].
    Problem(ProblemError),
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
            Self::InvalidTournamentSize(size) => {
                write!(f, "tournament size must be at least one (received {size})")
            }
            Self::SelectionFailed => f.write_str("selection operator failed to provide parents"),
            Self::Bounds(err) => write!(f, "{err}"),
            Self::Problem(err) => write!(f, "{err}"),
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
    P: Problem,
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

/// Real-coded genetic algorithm engine that operates over a [`Problem`].
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
    P: Problem,
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
        Self::update_best(&population, &fitness, &mut best_solution, &mut best_fitness);
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
            Self::update_best(&population, &fitness, &mut best_solution, &mut best_fitness);
        }
        Ok(RealGaReport {
            best_solution,
            best_fitness,
            generations: generation,
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
        let mut fitness = Vec::with_capacity(population.len());
        for candidate in population {
            fitness.push(self.problem.evaluate(candidate.as_slice())?);
        }
        Ok(fitness)
    }

    fn update_best(
        population: &[Vec<f64>],
        fitness: &[f64],
        best_solution: &mut Vec<f64>,
        best_fitness: &mut f64,
    ) {
        if let Some((idx, &value)) = fitness
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
        {
            if value < *best_fitness {
                *best_fitness = value;
                best_solution.clone_from(&population[idx]);
            }
        }
    }

    fn selection_scores(fitness: &[f64]) -> Vec<f64> {
        fitness.iter().map(|value| -*value).collect()
    }
}

/// Simulated Binary Crossover (SBX) implementation.
#[derive(Debug, Clone)]
pub struct SimulatedBinaryCrossover {
    distribution_index: f64,
}

impl SimulatedBinaryCrossover {
    /// Creates a new SBX operator.
    ///
    /// # Errors
    /// Returns [`RealGaError::InvalidDistributionIndex`] when the provided
    /// distribution index is non-positive or not finite.
    pub fn new(distribution_index: f64) -> Result<Self, RealGaError> {
        if !(distribution_index.is_finite() && distribution_index > 0.0) {
            return Err(RealGaError::InvalidDistributionIndex {
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
    /// Returns [`RealGaError`] when the distribution index or probability is
    /// invalid, or when the bound vectors have mismatched lengths.
    pub fn new(
        lower_bounds: Vec<f64>,
        upper_bounds: Vec<f64>,
        distribution_index: f64,
        probability: f64,
    ) -> Result<Self, RealGaError> {
        if !(distribution_index.is_finite() && distribution_index > 0.0) {
            return Err(RealGaError::InvalidDistributionIndex {
                operator: "polynomial mutation",
                value: distribution_index,
            });
        }
        if !(probability.is_finite() && (0.0..=1.0).contains(&probability)) {
            return Err(RealGaError::InvalidMutationProbability(probability));
        }
        if lower_bounds.len() != upper_bounds.len() {
            return Err(RealGaError::Bounds(
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

fn random_unit(rng: &mut dyn RngCore) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let value = rng.next_u64() as f64;
    value * RNG_SCALE
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
    use crate::ops::ProblemBounds;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

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
    }
}
