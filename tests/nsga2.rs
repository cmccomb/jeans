use jeans::ops::{ProblemBounds, ProblemResult};
use jeans::{MultiObjectiveProblem, Nsga2};
use rand::SeedableRng;

struct LinearFront;

impl ProblemBounds for LinearFront {
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

impl MultiObjectiveProblem for LinearFront {
    fn objectives(&self) -> usize {
        2
    }

    fn evaluate(&mut self, genes: &[f64]) -> ProblemResult<Vec<f64>> {
        Ok(vec![genes[0], 1.0 - genes[0]])
    }
}

#[test]
fn nsga2_finds_non_dominated_solutions() {
    let problem = LinearFront;
    let mut engine = Nsga2::builder(problem)
        .population_size(12)
        .generations(8)
        .build()
        .unwrap();
    let mut rng = rand::rngs::StdRng::seed_from_u64(13);
    let report = engine.run(&mut rng).unwrap();
    assert!(!report.pareto_objectives.is_empty());
    assert_eq!(report.generations, report.experiment.metadata.generations);
    assert_eq!(report.experiment.final_population.len(), 12);
    assert!(report.experiment.stats.generations() > 0);
    for (idx, objectives) in report.pareto_objectives.iter().enumerate() {
        for (other_idx, other) in report.pareto_objectives.iter().enumerate() {
            if idx == other_idx {
                continue;
            }
            assert!(
                !dominates(other, objectives),
                "solution {other_idx} dominates {idx}: {other:?} vs {objectives:?}",
                other_idx = other_idx,
                idx = idx,
                other = other,
                objectives = objectives,
            );
        }
    }
}

fn dominates(candidate: &[f64], other: &[f64]) -> bool {
    let mut strictly_better = false;
    for (&a, &b) in candidate.iter().zip(other.iter()) {
        if a > b {
            return false;
        }
        if a < b {
            strictly_better = true;
        }
    }
    strictly_better
}
