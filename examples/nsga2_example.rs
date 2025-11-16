use jeans::ops::{ProblemBounds, ProblemResult};
use jeans::{MultiObjectiveProblem, Nsga2};
use rand::SeedableRng;
use std::error::Error;

struct CantileverBeam {
    lower: [f64; 2],
    upper: [f64; 2],
    density: f64,
    youngs_modulus: f64,
    load: f64,
    length: f64,
}

impl CantileverBeam {
    fn new() -> Self {
        Self {
            lower: [0.05, 0.05],
            upper: [0.5, 0.5],
            density: 7850.0,
            youngs_modulus: 210e9,
            load: 1_000.0,
            length: 2.0,
        }
    }
}

impl ProblemBounds for CantileverBeam {
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

impl MultiObjectiveProblem for CantileverBeam {
    fn objectives(&self) -> usize {
        2
    }

    fn evaluate(&mut self, genes: &[f64]) -> ProblemResult<Vec<f64>> {
        let width = genes[0];
        let thickness = genes[1];
        let area = width * thickness;
        let volume = area * self.length;
        let mass = volume * self.density;
        let moment_of_inertia = width * thickness.powi(3) / 12.0;
        let deflection =
            (self.load * self.length.powi(3)) / (3.0 * self.youngs_modulus * moment_of_inertia);
        Ok(vec![mass, deflection])
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let problem = CantileverBeam::new();
    let mut engine = Nsga2::builder(problem)
        .population_size(80)
        .generations(75)
        .build()?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
    let report = engine.run(&mut rng)?;

    println!(
        "generated {} Pareto optimal solutions",
        report.pareto_solutions.len()
    );
    for (solution, objectives) in report
        .pareto_solutions
        .iter()
        .zip(report.pareto_objectives.iter())
        .take(5)
    {
        println!(
            "design {:?} => mass {:.2} kg, tip deflection {:.6} m",
            solution, objectives[0], objectives[1]
        );
    }

    Ok(())
}
