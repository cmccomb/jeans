use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use jeans::ops::{Problem, ProblemBounds, ProblemResult};
use jeans::{RealGa, StopCondition};
use rand::rngs::StdRng;
use rand::SeedableRng;

struct SphereProblem {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl SphereProblem {
    fn new(dimensions: usize) -> Self {
        Self {
            lower: vec![-5.12; dimensions],
            upper: vec![5.12; dimensions],
        }
    }
}

impl ProblemBounds for SphereProblem {
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

impl Problem for SphereProblem {
    fn evaluate(&mut self, genes: &[f64]) -> ProblemResult<f64> {
        self.validate_candidate_length(genes.len())?;
        Ok(genes.iter().map(|value| value * value).sum())
    }
}

fn run_ga(problem: SphereProblem, generations: usize, population: usize) {
    let mut engine = RealGa::builder(problem)
        .population_size(population)
        .stop_condition(StopCondition::max_generations(generations))
        .build()
        .expect("valid GA configuration");
    let mut rng = StdRng::seed_from_u64(42);
    engine.run(&mut rng).expect("optimization to succeed");
}

fn sphere_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("sphere-real-ga");
    for &dimensions in &[30_usize, 60_usize] {
        group.bench_function(BenchmarkId::from_parameter(dimensions), |b| {
            b.iter_batched(
                || SphereProblem::new(dimensions),
                |problem| run_ga(problem, 200, 200),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, sphere_benchmark);
criterion_main!(benches);
