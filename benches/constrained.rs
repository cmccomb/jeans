use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use jeans::ops::{Problem, ProblemBounds, ProblemResult};
use jeans::{RealGa, StopCondition};
use rand::rngs::StdRng;
use rand::SeedableRng;

struct ConstrainedQuadratic {
    lower: [f64; 2],
    upper: [f64; 2],
}

impl ConstrainedQuadratic {
    const fn new() -> Self {
        Self {
            lower: [0.0, 0.0],
            upper: [1.0, 1.0],
        }
    }
}

impl ProblemBounds for ConstrainedQuadratic {
    fn dimensions(&self) -> usize {
        2
    }

    fn lower_bounds(&self) -> &[f64] {
        &self.lower
    }

    fn upper_bounds(&self) -> &[f64] {
        &self.upper
    }
}

impl Problem for ConstrainedQuadratic {
    fn evaluate(&mut self, genes: &[f64]) -> ProblemResult<f64> {
        self.validate_candidate_length(genes.len())?;
        let objective = genes.iter().map(|value| value * value).sum::<f64>();
        let constraint = genes[0] + genes[1] - 1.0;
        let penalty = if constraint < 0.0 {
            constraint * constraint
        } else {
            0.0
        };
        Ok(objective + 100.0 * penalty)
    }
}

fn constrained_benchmark(c: &mut Criterion) {
    c.bench_function("constrained-real-ga", |b| {
        b.iter_batched(
            ConstrainedQuadratic::new,
            |problem| {
                let mut engine = RealGa::builder(problem)
                    .population_size(150)
                    .stop_condition(StopCondition::max_generations(250))
                    .build()
                    .expect("valid GA configuration");
                let mut rng = StdRng::seed_from_u64(9001);
                engine.run(&mut rng).expect("optimization to succeed");
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, constrained_benchmark);
criterion_main!(benches);
