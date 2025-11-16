//! NSGA-II implementation specialized for real-valued chromosomes.
//!
//! The [`Nsga2`] engine wires multi-objective problems with the SBX crossover
//! and polynomial mutation operators shipped with the crate. It exposes a
//! builder that mirrors [`crate::real_ga::RealGa`], allowing applications to
//! customize the population size, number of generations, and the underlying
//! operators.

use crate::ops::{
    CrossoverOperator, MultiObjectiveProblem, MutationOperator, OperatorError, PolynomialMutation,
    ProblemError, SimulatedBinaryCrossover,
};
use rand::distributions::Uniform;
use rand::{Rng, RngCore};
use std::fmt::{self, Display, Formatter};

const DEFAULT_SBX_ETA: f64 = 15.0;
const DEFAULT_POLY_ETA: f64 = 20.0;
const DEFAULT_GENERATIONS: usize = 100;

/// Report returned by [`Nsga2::run`].
///
/// # Examples
/// ```
/// use jeans::{MultiObjectiveProblem, Nsga2};
/// use jeans::ops::{ProblemBounds, ProblemResult};
/// use rand::SeedableRng;
///
/// struct LinearFront;
///
/// impl ProblemBounds for LinearFront {
///     fn dimensions(&self) -> usize { 1 }
///     fn lower_bounds(&self) -> &[f64] { &[0.0] }
///     fn upper_bounds(&self) -> &[f64] { &[1.0] }
/// }
///
/// impl MultiObjectiveProblem for LinearFront {
///     fn objectives(&self) -> usize { 2 }
///
///     fn evaluate(&mut self, genes: &[f64]) -> ProblemResult<Vec<f64>> {
///         Ok(vec![genes[0], 1.0 - genes[0]])
///     }
/// }
///
/// let problem = LinearFront;
/// let mut engine = Nsga2::builder(problem)
///     .population_size(10)
///     .generations(5)
///     .build()
///     .unwrap();
/// let mut rng = rand::rngs::StdRng::seed_from_u64(7);
/// let report = engine.run(&mut rng).unwrap();
/// assert!(!report.pareto_solutions.is_empty());
/// assert_eq!(report.pareto_solutions.len(), report.pareto_objectives.len());
/// ```
#[derive(Debug, Clone)]
pub struct Nsga2Report {
    /// Chromosomes found in the final Pareto front.
    pub pareto_solutions: Vec<Vec<f64>>,
    /// Objective values associated with [`Self::pareto_solutions`].
    pub pareto_objectives: Vec<Vec<f64>>,
    /// Number of generations executed by [`Nsga2::run`].
    pub generations: usize,
}

/// Errors produced by the [`Nsga2`] engine.
#[derive(Debug)]
pub enum Nsga2Error {
    /// Population size must be greater than zero.
    InvalidPopulationSize(usize),
    /// Number of generations must be at least one.
    InvalidGenerationCount(usize),
    /// Distribution index used by SBX or polynomial mutation was invalid.
    InvalidDistributionIndex {
        /// Operator reporting the invalid distribution index.
        operator: &'static str,
        /// Offending distribution index.
        value: f64,
    },
    /// Mutation probability was invalid.
    InvalidMutationProbability(f64),
    /// Operator parameter failed validation.
    InvalidOperatorParameter {
        /// Operator reporting the invalid parameter.
        operator: &'static str,
        /// Name of the invalid parameter.
        parameter: &'static str,
        /// Offending value.
        value: f64,
    },
    /// The problem reported inconsistent bounds.
    Bounds(crate::core::BoundsError),
    /// The problem reported an error.
    Problem(ProblemError),
    /// The problem returned a different number of objectives than advertised.
    ObjectiveCountMismatch {
        /// Number of objectives advertised by the problem.
        expected: usize,
        /// Number of objectives returned by the evaluator.
        found: usize,
    },
}

impl Display for Nsga2Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPopulationSize(size) => {
                write!(
                    f,
                    "population size must be greater than zero (received {size})"
                )
            }
            Self::InvalidGenerationCount(count) => {
                write!(
                    f,
                    "number of generations must be positive (received {count})"
                )
            }
            Self::InvalidDistributionIndex { operator, value } => {
                write!(
                    f,
                    "{operator} distribution index must be positive (received {value})"
                )
            }
            Self::InvalidMutationProbability(value) => {
                write!(
                    f,
                    "mutation probability must be within [0, 1] (received {value})"
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
            Self::Bounds(err) => write!(f, "{err}"),
            Self::Problem(err) => write!(f, "{err}"),
            Self::ObjectiveCountMismatch { expected, found } => {
                write!(
                    f,
                    "problem reported {found} objectives but advertised {expected}"
                )
            }
        }
    }
}

impl std::error::Error for Nsga2Error {}

impl From<ProblemError> for Nsga2Error {
    fn from(err: ProblemError) -> Self {
        Self::Problem(err)
    }
}

impl From<crate::core::BoundsError> for Nsga2Error {
    fn from(err: crate::core::BoundsError) -> Self {
        Self::Bounds(err)
    }
}

impl From<OperatorError> for Nsga2Error {
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

/// Builder used to configure an [`Nsga2`] engine.
pub struct Nsga2Builder<P> {
    problem: P,
    population_size: usize,
    generations: usize,
    crossover: Option<Box<dyn CrossoverOperator>>,
    mutation: Option<Box<dyn MutationOperator>>,
    lower_bounds: Vec<f64>,
    upper_bounds: Vec<f64>,
}

impl<P> Nsga2Builder<P>
where
    P: MultiObjectiveProblem,
{
    /// Configures the population size.
    #[must_use]
    pub fn population_size(mut self, size: usize) -> Self {
        self.population_size = size;
        self
    }

    /// Configures the number of generations.
    #[must_use]
    pub fn generations(mut self, generations: usize) -> Self {
        self.generations = generations;
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

    /// Finalizes the builder into an [`Nsga2`] engine.
    ///
    /// # Errors
    /// Returns [`Nsga2Error`] when the configuration is invalid or when the
    /// default operators fail validation.
    pub fn build(self) -> Result<Nsga2<P>, Nsga2Error> {
        if self.population_size == 0 {
            return Err(Nsga2Error::InvalidPopulationSize(0));
        }
        if self.generations == 0 {
            return Err(Nsga2Error::InvalidGenerationCount(0));
        }
        if self.lower_bounds.len() != self.upper_bounds.len() {
            return Err(Nsga2Error::Bounds(
                crate::core::BoundsError::DimensionMismatch {
                    expected: self.lower_bounds.len(),
                    found: self.upper_bounds.len(),
                },
            ));
        }
        let problem = self.problem;
        problem.validate_bounds()?;
        let dimensions = problem.dimensions();
        let objectives = problem.objectives();
        if objectives == 0 {
            return Err(Nsga2Error::ObjectiveCountMismatch {
                expected: 1,
                found: 0,
            });
        }
        let crossover: Box<dyn CrossoverOperator> = match self.crossover {
            Some(operator) => operator,
            None => Box::new(SimulatedBinaryCrossover::new(DEFAULT_SBX_ETA)?),
        };
        let mutation = if let Some(operator) = self.mutation {
            operator
        } else {
            #[allow(clippy::cast_precision_loss)]
            let probability = if dimensions == 0 {
                1.0
            } else {
                1.0 / dimensions as f64
            };
            Box::new(PolynomialMutation::new(
                self.lower_bounds.clone(),
                self.upper_bounds.clone(),
                DEFAULT_POLY_ETA,
                probability,
            )?)
        };
        Ok(Nsga2 {
            problem,
            population_size: self.population_size,
            generations: self.generations,
            crossover,
            mutation,
            lower_bounds: self.lower_bounds,
            upper_bounds: self.upper_bounds,
            objectives,
        })
    }
}

/// NSGA-II engine for multi-objective real-coded optimization.
pub struct Nsga2<P> {
    problem: P,
    population_size: usize,
    generations: usize,
    crossover: Box<dyn CrossoverOperator>,
    mutation: Box<dyn MutationOperator>,
    lower_bounds: Vec<f64>,
    upper_bounds: Vec<f64>,
    objectives: usize,
}

impl<P> Nsga2<P>
where
    P: MultiObjectiveProblem,
{
    /// Creates a builder used to configure the engine.
    #[must_use]
    pub fn builder(problem: P) -> Nsga2Builder<P> {
        let lower_bounds = problem.lower_bounds().to_vec();
        let upper_bounds = problem.upper_bounds().to_vec();
        Nsga2Builder {
            problem,
            population_size: 100,
            generations: DEFAULT_GENERATIONS,
            crossover: None,
            mutation: None,
            lower_bounds,
            upper_bounds,
        }
    }

    /// Runs the optimization for the configured number of generations.
    ///
    /// # Errors
    /// Returns [`Nsga2Error`] when the problem evaluation fails.
    pub fn run<R: Rng>(&mut self, rng: &mut R) -> Result<Nsga2Report, Nsga2Error> {
        let mut population = self.initialize_population(rng);
        self.evaluate_population(&mut population)?;
        Self::assign_ranks_and_crowding(&mut population);
        for _ in 0..self.generations {
            let mut offspring = self.generate_offspring(&population, rng);
            self.evaluate_population(&mut offspring)?;
            let mut combined = population;
            combined.append(&mut offspring);
            population = self.reduce_population(combined);
        }
        Self::assign_ranks_and_crowding(&mut population);
        let pareto_front = Self::front_indices(&population, 0);
        let pareto_solutions = pareto_front
            .iter()
            .map(|&idx| population[idx].genes.clone())
            .collect();
        let pareto_objectives = pareto_front
            .iter()
            .map(|&idx| population[idx].objectives.clone())
            .collect();
        Ok(Nsga2Report {
            pareto_solutions,
            pareto_objectives,
            generations: self.generations,
        })
    }

    fn initialize_population<R: Rng>(&self, rng: &mut R) -> Vec<IndividualState> {
        let mut population = Vec::with_capacity(self.population_size);
        for _ in 0..self.population_size {
            population.push(IndividualState::from_genes(self.random_candidate(rng)));
        }
        population
    }

    fn random_candidate<R: Rng>(&self, rng: &mut R) -> Vec<f64> {
        let mut genes = Vec::with_capacity(self.lower_bounds.len());
        for (&lower, &upper) in self.lower_bounds.iter().zip(self.upper_bounds.iter()) {
            let sampler = Uniform::new_inclusive(lower, upper);
            let value = rng.sample(sampler);
            genes.push(value);
        }
        genes
    }

    fn evaluate_population(
        &mut self,
        population: &mut [IndividualState],
    ) -> Result<(), Nsga2Error> {
        for individual in population {
            let objectives = self.problem.evaluate(individual.genes.as_slice())?;
            if objectives.len() != self.objectives {
                return Err(Nsga2Error::ObjectiveCountMismatch {
                    expected: self.objectives,
                    found: objectives.len(),
                });
            }
            individual.objectives = objectives;
        }
        Ok(())
    }

    fn generate_offspring<R: Rng>(
        &self,
        population: &[IndividualState],
        rng: &mut R,
    ) -> Vec<IndividualState> {
        let mut offspring = Vec::with_capacity(self.population_size);
        while offspring.len() < self.population_size {
            let parent_a = &population[Self::binary_tournament(population, rng)];
            let parent_b = &population[Self::binary_tournament(population, rng)];
            let (raw_child_a, raw_child_b) =
                self.crossover
                    .crossover(&parent_a.genes, &parent_b.genes, rng);
            let mut child_a = self.mutation.mutate(raw_child_a.as_slice(), rng);
            let mut child_b = self.mutation.mutate(raw_child_b.as_slice(), rng);
            self.clamp(&mut child_a);
            self.clamp(&mut child_b);
            offspring.push(IndividualState::from_genes(child_a));
            if offspring.len() < self.population_size {
                offspring.push(IndividualState::from_genes(child_b));
            }
        }
        offspring
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

    fn reduce_population(&self, mut combined: Vec<IndividualState>) -> Vec<IndividualState> {
        let fronts = Self::assign_ranks_and_crowding(&mut combined);
        let mut next = Vec::with_capacity(self.population_size);
        for mut front in fronts {
            if next.len() == self.population_size {
                break;
            }
            if next.len() + front.len() <= self.population_size {
                for idx in front {
                    next.push(combined[idx].clone());
                }
                continue;
            }
            front.sort_by(|&a, &b| {
                combined[b]
                    .crowding_distance
                    .total_cmp(&combined[a].crowding_distance)
            });
            for idx in front.into_iter().take(self.population_size - next.len()) {
                next.push(combined[idx].clone());
            }
        }
        next
    }

    fn assign_ranks_and_crowding(population: &mut [IndividualState]) -> Vec<Vec<usize>> {
        let fronts = Self::fast_nondominated_sort(population);
        for front in &fronts {
            Self::assign_crowding_distance(population, front);
        }
        fronts
    }

    fn fast_nondominated_sort(population: &mut [IndividualState]) -> Vec<Vec<usize>> {
        let size = population.len();
        let mut domination_counts = vec![0usize; size];
        let mut dominated = vec![Vec::new(); size];
        let mut fronts: Vec<Vec<usize>> = Vec::new();
        let mut current_front = Vec::new();
        for p in 0..size {
            dominated[p].clear();
            domination_counts[p] = 0;
            for q in 0..size {
                if dominates(&population[p].objectives, &population[q].objectives) {
                    dominated[p].push(q);
                } else if dominates(&population[q].objectives, &population[p].objectives) {
                    domination_counts[p] += 1;
                }
            }
            if domination_counts[p] == 0 {
                population[p].rank = 0;
                current_front.push(p);
            }
        }
        let mut rank = 0usize;
        let mut next_front;
        while !current_front.is_empty() {
            fronts.push(current_front.clone());
            next_front = Vec::new();
            for &p in &fronts[rank] {
                for &q in &dominated[p] {
                    domination_counts[q] -= 1;
                    if domination_counts[q] == 0 {
                        population[q].rank = rank + 1;
                        next_front.push(q);
                    }
                }
            }
            rank += 1;
            current_front = next_front;
        }
        fronts
    }

    fn assign_crowding_distance(population: &mut [IndividualState], front: &[usize]) {
        if front.is_empty() {
            return;
        }
        if front.len() <= 2 {
            for &idx in front {
                population[idx].crowding_distance = f64::INFINITY;
            }
            return;
        }
        for &idx in front {
            population[idx].crowding_distance = 0.0;
        }
        let objectives = population[front[0]].objectives.len();
        for obj_idx in 0..objectives {
            let mut sorted = front.to_vec();
            sorted.sort_by(|&a, &b| {
                population[a].objectives[obj_idx].total_cmp(&population[b].objectives[obj_idx])
            });
            let first = sorted[0];
            let last = sorted[sorted.len() - 1];
            population[first].crowding_distance = f64::INFINITY;
            population[last].crowding_distance = f64::INFINITY;
            let min = population[first].objectives[obj_idx];
            let max = population[last].objectives[obj_idx];
            if (max - min).abs() < f64::EPSILON {
                continue;
            }
            for window in sorted.windows(3) {
                if let [prev, current, next] = window {
                    let distance = (population[*next].objectives[obj_idx]
                        - population[*prev].objectives[obj_idx])
                        / (max - min);
                    if population[*current].crowding_distance.is_finite() {
                        population[*current].crowding_distance += distance;
                    }
                }
            }
        }
    }

    fn binary_tournament<R: Rng>(population: &[IndividualState], rng: &mut R) -> usize {
        if population.len() == 1 {
            return 0;
        }
        let idx_a = random_index(population.len(), rng);
        let mut idx_b = random_index(population.len(), rng);
        while idx_a == idx_b {
            idx_b = random_index(population.len(), rng);
        }
        if compare_rank_and_distance(&population[idx_a], &population[idx_b])
            == std::cmp::Ordering::Greater
        {
            idx_b
        } else {
            idx_a
        }
    }

    fn front_indices(population: &[IndividualState], rank: usize) -> Vec<usize> {
        population
            .iter()
            .enumerate()
            .filter_map(|(idx, individual)| (individual.rank == rank).then_some(idx))
            .collect()
    }
}

#[derive(Clone)]
struct IndividualState {
    genes: Vec<f64>,
    objectives: Vec<f64>,
    rank: usize,
    crowding_distance: f64,
}

impl IndividualState {
    fn from_genes(genes: Vec<f64>) -> Self {
        Self {
            genes,
            objectives: Vec::new(),
            rank: 0,
            crowding_distance: 0.0,
        }
    }
}

fn dominates(candidate: &[f64], other: &[f64]) -> bool {
    let mut strictly_better = false;
    for (&a, &b) in candidate.iter().zip(other.iter()) {
        if a > b {
            return false;
        }
        if a < b {
            strictly_better = true;
        }
    }
    strictly_better
}

fn compare_rank_and_distance(a: &IndividualState, b: &IndividualState) -> std::cmp::Ordering {
    match a.rank.cmp(&b.rank) {
        std::cmp::Ordering::Less => std::cmp::Ordering::Less,
        std::cmp::Ordering::Greater => std::cmp::Ordering::Greater,
        std::cmp::Ordering::Equal => b.crowding_distance.total_cmp(&a.crowding_distance),
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
    use crate::ops::{ProblemBounds, ProblemResult};
    use rand::SeedableRng;

    #[derive(Clone)]
    struct TwoObjectives;

    impl ProblemBounds for TwoObjectives {
        fn dimensions(&self) -> usize {
            1
        }

        fn lower_bounds(&self) -> &[f64] {
            &[-1.0]
        }

        fn upper_bounds(&self) -> &[f64] {
            &[1.0]
        }
    }

    impl MultiObjectiveProblem for TwoObjectives {
        fn objectives(&self) -> usize {
            2
        }

        fn evaluate(&mut self, genes: &[f64]) -> ProblemResult<Vec<f64>> {
            Ok(vec![genes[0] * genes[0], (genes[0] - 1.0).abs()])
        }
    }

    #[test]
    fn dominates_detects_strict_improvement() {
        assert!(dominates(&[0.0, 0.5], &[0.1, 0.5]));
        assert!(!dominates(&[0.1, 0.5], &[0.0, 0.5]));
    }

    #[test]
    fn crowding_distance_marks_edges() {
        let mut population = vec![
            IndividualState {
                genes: vec![],
                objectives: vec![0.0],
                rank: 0,
                crowding_distance: 0.0,
            },
            IndividualState {
                genes: vec![],
                objectives: vec![1.0],
                rank: 0,
                crowding_distance: 0.0,
            },
            IndividualState {
                genes: vec![],
                objectives: vec![0.5],
                rank: 0,
                crowding_distance: 0.0,
            },
        ];
        Nsga2::<TwoObjectives>::assign_crowding_distance(&mut population, &[0, 1, 2]);
        assert!(population[0].crowding_distance.is_infinite());
        assert!(population[1].crowding_distance.is_infinite());
        assert!(population[2].crowding_distance.is_finite());
    }

    #[test]
    fn nsga2_runs_and_returns_front() {
        let problem = TwoObjectives;
        let mut engine = Nsga2::builder(problem)
            .population_size(8)
            .generations(5)
            .build()
            .unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        let report = engine.run(&mut rng).unwrap();
        assert!(!report.pareto_solutions.is_empty());
        assert_eq!(
            report.pareto_solutions.len(),
            report.pareto_objectives.len()
        );
    }
}
