//! Asynchronous evaluation helpers for genetic algorithm engines.
//!
//! The types in this module bridge [`AsyncProblem`]
//! implementations with the synchronous engines shipped by the crate. They
//! spawn Tokio tasks to evaluate fitness in parallel and expose a
//! [`SingleObjectiveEvaluator`] trait that both synchronous and asynchronous
//! problems can implement, allowing engines to switch between evaluation modes
//! without changing their core logic.

use crate::ops::{AsyncProblem, Problem, ProblemBounds, ProblemError, ProblemResult};
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::sync::Arc;
use std::thread;
use tokio::runtime::{Builder, Runtime};
use tokio::task::JoinHandle;

/// Result type produced by [`SingleObjectiveEvaluator`] implementations.
pub type EvaluationResult<T> = Result<T, EvaluationError>;

/// Errors reported by evaluation backends.
#[derive(Debug)]
pub enum EvaluationError {
    /// Wrapper around [`ProblemError`].
    Problem(ProblemError),
    /// Tokio runtime failed to initialize.
    Runtime(std::io::Error),
    /// A spawned task failed or panicked before producing a fitness value.
    Task(tokio::task::JoinError),
}

impl Display for EvaluationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Problem(err) => write!(f, "{err}"),
            Self::Runtime(err) => write!(f, "failed to initialize Tokio runtime: {err}"),
            Self::Task(err) => write!(f, "Tokio task failed: {err}"),
        }
    }
}

impl Error for EvaluationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Problem(err) => Some(err),
            Self::Runtime(err) => Some(err),
            Self::Task(err) => Some(err),
        }
    }
}

impl From<ProblemError> for EvaluationError {
    fn from(err: ProblemError) -> Self {
        Self::Problem(err)
    }
}

impl From<tokio::task::JoinError> for EvaluationError {
    fn from(err: tokio::task::JoinError) -> Self {
        Self::Task(err)
    }
}

/// Trait implemented by types that can evaluate populations of chromosomes.
pub trait SingleObjectiveEvaluator: ProblemBounds {
    /// Evaluates the provided population and returns their fitness scores.
    ///
    /// # Errors
    /// Implementations may return [`EvaluationError`] when the underlying
    /// problem reports [`ProblemError`] or when task execution fails.
    fn evaluate_population(&mut self, population: &[Vec<f64>]) -> EvaluationResult<Vec<f64>>;
}

impl<T> SingleObjectiveEvaluator for T
where
    T: Problem,
{
    fn evaluate_population(&mut self, population: &[Vec<f64>]) -> EvaluationResult<Vec<f64>> {
        let mut fitness = Vec::with_capacity(population.len());
        for candidate in population {
            fitness.push(self.evaluate(candidate.as_slice())?);
        }
        Ok(fitness)
    }
}

/// Errors that can occur while building an [`AsyncBatchEvaluator`].
#[derive(Debug)]
pub enum AsyncEvaluatorError {
    /// The requested concurrency level was zero.
    InvalidConcurrency,
    /// Tokio runtime initialization failed.
    Runtime(std::io::Error),
}

impl Display for AsyncEvaluatorError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConcurrency => {
                write!(
                    f,
                    "max concurrency must be at least one for async evaluation"
                )
            }
            Self::Runtime(err) => write!(f, "failed to initialize Tokio runtime: {err}"),
        }
    }
}

impl Error for AsyncEvaluatorError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidConcurrency => None,
            Self::Runtime(err) => Some(err),
        }
    }
}

/// Evaluates [`AsyncProblem`] implementations in parallel Tokio tasks.
pub struct AsyncBatchEvaluator<P>
where
    P: AsyncProblem + 'static,
{
    problem: Arc<P>,
    runtime: Runtime,
    max_tasks: usize,
}

impl<P> AsyncBatchEvaluator<P>
where
    P: AsyncProblem + 'static,
{
    /// Creates a batch evaluator with a concurrency level that matches the
    /// available parallelism on the current machine.
    ///
    /// # Errors
    /// Returns [`AsyncEvaluatorError::Runtime`] when the Tokio runtime cannot
    /// be initialized.
    pub fn new(problem: P) -> Result<Self, AsyncEvaluatorError> {
        let parallelism = thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1);
        Self::with_max_concurrency(problem, parallelism)
    }

    /// Creates a batch evaluator with the requested maximum number of tasks.
    ///
    /// # Errors
    /// Returns [`AsyncEvaluatorError::InvalidConcurrency`] when `max_tasks` is
    /// zero or [`AsyncEvaluatorError::Runtime`] if the Tokio runtime fails to
    /// initialize.
    pub fn with_max_concurrency(problem: P, max_tasks: usize) -> Result<Self, AsyncEvaluatorError> {
        if max_tasks == 0 {
            return Err(AsyncEvaluatorError::InvalidConcurrency);
        }
        let runtime = Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(AsyncEvaluatorError::Runtime)?;
        Ok(Self {
            problem: Arc::new(problem),
            runtime,
            max_tasks,
        })
    }

    async fn evaluate_batch(&self, population: &[Vec<f64>]) -> EvaluationResult<Vec<f64>> {
        let mut pending: Vec<JoinHandle<ProblemResult<(usize, f64)>>> = Vec::new();
        let mut fitness = vec![0.0_f64; population.len()];
        for (idx, candidate) in population.iter().enumerate() {
            let problem = Arc::clone(&self.problem);
            let genes = candidate.clone();
            pending.push(tokio::spawn(async move {
                let score = problem.evaluate_async(genes.as_slice()).await?;
                Ok((idx, score))
            }));
            if pending.len() >= self.max_tasks {
                Self::resolve_handles(&mut pending, &mut fitness).await?;
            }
        }
        Self::resolve_handles(&mut pending, &mut fitness).await?;
        Ok(fitness)
    }

    async fn resolve_handles(
        pending: &mut Vec<JoinHandle<ProblemResult<(usize, f64)>>>,
        fitness: &mut [f64],
    ) -> EvaluationResult<()> {
        while let Some(handle) = pending.pop() {
            let (idx, score) = handle.await??;
            fitness[idx] = score;
        }
        Ok(())
    }
}

impl<P> ProblemBounds for AsyncBatchEvaluator<P>
where
    P: AsyncProblem + 'static,
{
    fn dimensions(&self) -> usize {
        self.problem.dimensions()
    }

    fn lower_bounds(&self) -> &[f64] {
        self.problem.lower_bounds()
    }

    fn upper_bounds(&self) -> &[f64] {
        self.problem.upper_bounds()
    }
}

impl<P> SingleObjectiveEvaluator for AsyncBatchEvaluator<P>
where
    P: AsyncProblem + 'static,
{
    fn evaluate_population(&mut self, population: &[Vec<f64>]) -> EvaluationResult<Vec<f64>> {
        self.runtime.block_on(self.evaluate_batch(population))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    struct TestAsyncProblem {
        lower: Vec<f64>,
        upper: Vec<f64>,
        calls: Arc<Mutex<Vec<usize>>>,
    }

    impl TestAsyncProblem {
        fn new(dimensions: usize) -> Self {
            Self {
                lower: vec![0.0; dimensions],
                upper: vec![1.0; dimensions],
                calls: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    impl ProblemBounds for TestAsyncProblem {
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

    #[async_trait]
    impl AsyncProblem for TestAsyncProblem {
        async fn evaluate_async(&self, genes: &[f64]) -> ProblemResult<f64> {
            tokio::time::sleep(Duration::from_millis(10)).await;
            let mut guard = self.calls.lock().expect("lock poisoned");
            guard.push(genes.len());
            Ok(genes.iter().sum())
        }
    }

    #[test]
    fn synchronous_problems_implement_evaluator_trait() {
        struct SyncProblem;

        impl ProblemBounds for SyncProblem {
            fn dimensions(&self) -> usize {
                2
            }

            fn lower_bounds(&self) -> &[f64] {
                &[-1.0, -1.0]
            }

            fn upper_bounds(&self) -> &[f64] {
                &[1.0, 1.0]
            }
        }

        impl Problem for SyncProblem {
            fn evaluate(&mut self, genes: &[f64]) -> ProblemResult<f64> {
                Ok(genes.iter().product())
            }
        }

        let mut problem = SyncProblem;
        let population = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let scores = problem.evaluate_population(&population).unwrap();
        assert_eq!(scores, vec![2.0, 12.0]);
    }

    #[test]
    fn async_batch_evaluator_runs_in_parallel() {
        let problem = TestAsyncProblem::new(2);
        let calls = Arc::clone(&problem.calls);
        let mut evaluator = AsyncBatchEvaluator::with_max_concurrency(problem, 2).unwrap();
        let population = vec![vec![1.0, 1.0], vec![2.0, 2.0], vec![3.0, 3.0]];
        let scores = evaluator.evaluate_population(&population).unwrap();
        assert_eq!(scores, vec![2.0, 4.0, 6.0]);
        let call_count = calls.lock().unwrap().len();
        assert_eq!(call_count, 3);
    }
}
