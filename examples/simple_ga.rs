use jeans::ops::{
    PolynomialMutation, Problem, ProblemBounds, ProblemResult, SimulatedBinaryCrossover,
};
use jeans::{RealGa, StopCondition};
use rand::SeedableRng;
use std::convert::TryFrom;
use std::error::Error;

struct Sphere {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl Sphere {
    fn new(dimensions: usize, bound: f64) -> Self {
        let lower = vec![-bound; dimensions];
        let upper = vec![bound; dimensions];
        Self { lower, upper }
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
    fn evaluate(&mut self, genes: &[f64]) -> ProblemResult<f64> {
        Ok(genes.iter().map(|value| value * value).sum())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let problem = Sphere::new(5, 5.12);
    let crossover = SimulatedBinaryCrossover::new(15.0)?;
    let dims = u32::try_from(problem.dimensions()).expect("dimensions must fit into u32");
    let mutation_probability = if dims == 0 {
        1.0
    } else {
        1.0 / f64::from(dims)
    };
    let mutation = PolynomialMutation::new(
        problem.lower_bounds().to_vec(),
        problem.upper_bounds().to_vec(),
        20.0,
        mutation_probability,
    )?;
    let stop_condition =
        StopCondition::target_fitness_below(1e-6).or(StopCondition::max_generations(250));

    let mut engine = RealGa::builder(problem)
        .population_size(80)
        .crossover(crossover)
        .mutation(mutation)
        .stop_condition(stop_condition)
        .build()?;

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let report = engine.run(&mut rng)?;
    println!(
        "best candidate: {:?} => {:.6}",
        report.best_solution, report.best_fitness
    );
    Ok(())
}
