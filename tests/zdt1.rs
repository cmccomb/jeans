use benchfun::{MultiObjective, Zdt1};
use jeans::ops::{MultiObjectiveProblem, ProblemBounds, ProblemResult};
use jeans::Nsga2;
use rand::SeedableRng;

struct Zdt1Problem;

impl ProblemBounds for Zdt1Problem {
    fn dimensions(&self) -> usize {
        3
    }

    fn lower_bounds(&self) -> &[f64] {
        &[0.0; 3]
    }

    fn upper_bounds(&self) -> &[f64] {
        &[1.0; 3]
    }
}

impl MultiObjectiveProblem for Zdt1Problem {
    fn objectives(&self) -> usize {
        2
    }

    fn evaluate(&mut self, genes: &[f64]) -> ProblemResult<Vec<f64>> {
        self.validate_candidate_length(genes.len())?;
        Ok(Zdt1::f(genes.to_vec()))
    }
}

#[test]
fn nsga2_approaches_the_shared_zdt1_reference_front() {
    let mut engine = Nsga2::builder(Zdt1Problem)
        .population_size(64)
        .generations(80)
        .build()
        .unwrap();
    let mut rng = rand::rngs::StdRng::seed_from_u64(777);
    let report = engine.run(&mut rng).unwrap();
    assert!(report.pareto_solutions.len() >= 10);
    for (genes, objectives) in report
        .pareto_solutions
        .iter()
        .zip(&report.pareto_objectives)
    {
        assert!(genes.iter().all(|value| (0.0..=1.0).contains(value)));
        assert_eq!(*objectives, Zdt1::f(genes.clone()));
        let reference = Zdt1::pareto_front(objectives[0]);
        let gap = objectives[1] - reference[1];
        // A generous convergence bound catches objective/integration regressions
        // without requiring a specific population or exact floating-point result.
        assert!((-1e-12..0.2).contains(&gap), "front gap: {gap}");
    }
}
