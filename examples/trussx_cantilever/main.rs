//! Demonstrates optimizing a cantilever truss with `jeans` and `trussx`.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example trussx_cantilever
//! ```

use jeans::{RealGa, StopCondition};
use rand::SeedableRng;
use std::error::Error;

mod problem;
use problem::CantileverSizingProblem;

fn main() -> Result<(), Box<dyn Error>> {
    let problem = CantileverSizingProblem::new();
    let stop_condition =
        StopCondition::target_fitness_below(1_000.0).or(StopCondition::max_generations(150));

    let mut engine = RealGa::builder(problem)
        .population_size(60)
        .stop_condition(stop_condition)
        .build()?;

    let mut rng = rand::rngs::StdRng::seed_from_u64(2024);
    let report = engine.run(&mut rng)?;

    let analysis = CantileverSizingProblem::new();
    let (_, tip_deflection) = analysis
        .analyze_with_deflection(&report.best_solution)
        .unwrap_or((report.best_fitness, 0.0));

    println!(
        "best areas: {:?}\nfitness: {:.3}\ntip deflection: {:.4} m",
        report.best_solution, report.best_fitness, tip_deflection
    );
    Ok(())
}
