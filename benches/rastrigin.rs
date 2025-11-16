use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use jeans::ops::{Problem, ProblemBounds, ProblemResult};
use jeans::{RealGa, StopCondition};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::f64::consts::PI;

struct RastriginProblem {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl RastriginProblem {
    fn new(dimensions: usize) -> Self {
        Self {
            lower: vec![-5.12; dimensions],
            upper: vec![5.12; dimensions],
        }
    }
}

impl ProblemBounds for RastriginProblem {
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

impl Problem for RastriginProblem {
    fn evaluate(&mut self, genes: &[f64]) -> ProblemResult<f64> {
        self.validate_candidate_length(genes.len())?;
        let a = 10.0;
        #[allow(clippy::cast_precision_loss)]
        let dims = genes.len() as f64;
        let sum = genes
            .iter()
            .map(|value| value * value - a * (2.0 * PI * value).cos())
            .sum::<f64>();
        Ok(a * dims + sum)
    }
}

fn run_ga(problem: RastriginProblem, generations: usize, population: usize) {
    let mut engine = RealGa::builder(problem)
        .population_size(population)
        .stop_condition(StopCondition::max_generations(generations))
        .build()
        .expect("valid GA configuration");
    let mut rng = StdRng::seed_from_u64(1337);
    engine.run(&mut rng).expect("optimization to succeed");
}

fn rastrigin_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("rastrigin-real-ga");
    for &dimensions in &[10_usize, 30_usize] {
        group.bench_function(BenchmarkId::from_parameter(dimensions), |b| {
            b.iter_batched(
                || RastriginProblem::new(dimensions),
                |problem| run_ga(problem, 300, 250),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, rastrigin_benchmark);
criterion_main!(benches);
