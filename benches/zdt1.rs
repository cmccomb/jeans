use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use jeans::ops::{MultiObjectiveProblem, ProblemBounds, ProblemError, ProblemResult};
use jeans::Nsga2;
use rand::rngs::StdRng;
use rand::SeedableRng;

struct Zdt1 {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl Zdt1 {
    fn new(dimensions: usize) -> Self {
        assert!(
            dimensions >= 2,
            "ZDT1 requires at least two decision variables"
        );
        Self {
            lower: vec![0.0; dimensions],
            upper: vec![1.0; dimensions],
        }
    }
}

impl ProblemBounds for Zdt1 {
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

impl MultiObjectiveProblem for Zdt1 {
    fn objectives(&self) -> usize {
        2
    }

    fn evaluate(&mut self, genes: &[f64]) -> ProblemResult<Vec<f64>> {
        self.validate_candidate_length(genes.len())?;
        let f1 = genes[0];
        let sum_tail: f64 = genes.iter().skip(1).sum();
        let dims = self.dimensions();
        #[allow(clippy::cast_precision_loss)]
        let denominator = (dims - 1) as f64;
        if denominator == 0.0 {
            return Err(ProblemError::DimensionMismatch {
                expected: 2,
                found: dims,
            });
        }
        let g = 1.0 + (9.0 / denominator) * sum_tail;
        let h = 1.0 - (f1 / g).sqrt();
        Ok(vec![f1, g * h])
    }
}

fn zdt1_benchmark(c: &mut Criterion) {
    c.bench_function("zdt1-nsga2", |b| {
        b.iter_batched(
            || Zdt1::new(30),
            |problem| {
                let mut engine = Nsga2::builder(problem)
                    .population_size(200)
                    .generations(250)
                    .build()
                    .expect("valid NSGA-II configuration");
                let mut rng = StdRng::seed_from_u64(777);
                engine.run(&mut rng).expect("optimization to succeed");
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, zdt1_benchmark);
criterion_main!(benches);
