use async_trait::async_trait;
use jeans::ops::{AsyncProblem, ProblemBounds, ProblemResult};
use jeans::Chromosome;
use rand::distributions::Uniform;
use rand::{Rng, SeedableRng};
use std::error::Error;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;

struct DelayedSphere {
    lower: Vec<f64>,
    upper: Vec<f64>,
    delay: Duration,
}

impl DelayedSphere {
    fn new(dimensions: usize, bound: f64, delay: Duration) -> Self {
        let lower = vec![-bound; dimensions];
        let upper = vec![bound; dimensions];
        Self {
            lower,
            upper,
            delay,
        }
    }
}

impl ProblemBounds for DelayedSphere {
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
impl AsyncProblem for DelayedSphere {
    async fn evaluate_async(&self, genes: &[f64]) -> ProblemResult<f64> {
        sleep(self.delay).await;
        Ok(genes.iter().map(|value| value * value).sum())
    }
}

type ExampleResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExampleResult<()> {
    let problem = Arc::new(DelayedSphere::new(4, 5.0, Duration::from_millis(75)));
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let candidates: Vec<Vec<f64>> = (0..8)
        .map(|_| random_candidate(&mut rng, &problem))
        .collect();

    let mut tasks = Vec::new();
    for genes in candidates {
        let chromosome = Chromosome::new(genes);
        let evaluator = Arc::clone(&problem);
        tasks.push(tokio::spawn(async move {
            let fitness = evaluator.evaluate_chromosome_async(&chromosome).await?;
            Ok::<_, jeans::ProblemError>((chromosome, fitness))
        }));
    }

    let mut best: Option<(Vec<f64>, f64)> = None;
    for task in tasks {
        let (chromosome, fitness) = task.await??;
        let genes = chromosome.genes().to_vec();
        println!("candidate {genes:?} => {fitness:.4}");
        match best {
            Some((_, best_fitness)) if fitness >= best_fitness => {}
            _ => best = Some((genes, fitness)),
        }
    }

    if let Some((genes, fitness)) = best {
        println!("fastest asynchronous run found {genes:?} with fitness {fitness:.4}");
    }

    Ok(())
}

fn random_candidate(rng: &mut rand::rngs::StdRng, problem: &DelayedSphere) -> Vec<f64> {
    problem
        .lower_bounds()
        .iter()
        .zip(problem.upper_bounds().iter())
        .map(|(&lower, &upper)| {
            let sampler = Uniform::new_inclusive(lower, upper);
            rng.sample(sampler)
        })
        .collect()
}
