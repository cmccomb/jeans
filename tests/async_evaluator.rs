use async_trait::async_trait;
use jeans::r#async::{AsyncBatchEvaluator, AsyncEvaluatorError, EvaluationError};
use jeans::{AsyncProblem, ProblemBounds, ProblemResult, SingleObjectiveEvaluator};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::time::{sleep, Duration};

struct OrderedAsyncProblem {
    lower: Vec<f64>,
    upper: Vec<f64>,
    delays: Vec<Duration>,
    in_flight: Arc<AtomicUsize>,
    max_in_flight: Arc<AtomicUsize>,
}

impl OrderedAsyncProblem {
    fn new(delays: Vec<Duration>) -> (Self, Arc<AtomicUsize>) {
        let in_flight = Arc::new(AtomicUsize::new(0));
        let max_in_flight = Arc::new(AtomicUsize::new(0));
        let problem = Self {
            lower: vec![0.0; 2],
            upper: vec![10.0; 2],
            delays,
            in_flight: Arc::clone(&in_flight),
            max_in_flight: Arc::clone(&max_in_flight),
        };
        (problem, max_in_flight)
    }
}

impl ProblemBounds for OrderedAsyncProblem {
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
impl AsyncProblem for OrderedAsyncProblem {
    async fn evaluate_async(&self, genes: &[f64]) -> ProblemResult<f64> {
        let current = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_in_flight.fetch_max(current, Ordering::SeqCst);
        let delay_idx = genes[0] as usize % self.delays.len();
        sleep(self.delays[delay_idx]).await;
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        Ok(genes.iter().sum())
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn preserves_population_order_with_concurrency() {
    let delays = vec![
        Duration::from_millis(30),
        Duration::from_millis(5),
        Duration::from_millis(20),
    ];
    let (problem, max_in_flight) = OrderedAsyncProblem::new(delays);
    let mut evaluator = AsyncBatchEvaluator::with_max_concurrency(problem, 2).unwrap();
    let population = vec![
        vec![0.0, 10.0],
        vec![1.0, 20.0],
        vec![2.0, 30.0],
        vec![0.0, 40.0],
    ];

    let scores = tokio::task::spawn_blocking(move || evaluator.evaluate_population(&population))
        .await
        .unwrap()
        .unwrap();

    assert_eq!(scores, vec![10.0, 21.0, 32.0, 40.0]);
    assert!(max_in_flight.load(Ordering::SeqCst) <= 2);
}

struct PanickingAsyncProblem;

impl ProblemBounds for PanickingAsyncProblem {
    fn dimensions(&self) -> usize {
        1
    }

    fn lower_bounds(&self) -> &[f64] {
        &[0.0]
    }

    fn upper_bounds(&self) -> &[f64] {
        &[1.0]
    }
}

#[async_trait]
impl AsyncProblem for PanickingAsyncProblem {
    async fn evaluate_async(&self, _genes: &[f64]) -> ProblemResult<f64> {
        panic!("forced panic for testing");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn panicking_task_surfaces_as_error() {
    let problem = PanickingAsyncProblem;
    let mut evaluator = AsyncBatchEvaluator::with_max_concurrency(problem, 1).unwrap();

    let result = tokio::task::spawn_blocking(move || evaluator.evaluate_population(&[vec![1.0]]))
        .await
        .unwrap();

    match result {
        Err(EvaluationError::Task(join_error)) => {
            assert!(join_error.is_panic());
        }
        other => panic!("unexpected result: {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn rejects_zero_concurrency() {
    let problem = OrderedAsyncProblem::new(vec![Duration::from_millis(1)]).0;
    match AsyncBatchEvaluator::with_max_concurrency(problem, 0) {
        Err(AsyncEvaluatorError::InvalidConcurrency) => {}
        Err(AsyncEvaluatorError::Runtime(io_error)) => {
            panic!("unexpected runtime error: {io_error}")
        }
        Ok(_) => panic!("zero concurrency should not construct an evaluator"),
    }
}
