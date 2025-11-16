![CI](https://github.com/cmccomb/jeans/actions/workflows/ci.yml/badge.svg?branch=master)
# About

This is a crate for implementing genetic algorithms. Specifically, this is for algorithms whose solutions can be represented as a vector of floating point values, but that might change in the future.

## Real-coded genetic algorithms

The [`RealGa`](https://docs.rs/jeans/latest/jeans/struct.RealGa.html) engine exposes a
builder-style API that wires common real-coded GA operators together. It
supports configuring the dimensionality/bounds via the provided [`Problem`],
population size, SBX crossover, polynomial mutation, tournament selection, and
flexible stop conditions. A minimal example looks like:

```rust
use jeans::{RealGa, StopCondition};
use jeans::ops::{Problem, ProblemBounds, ProblemResult};
use rand::SeedableRng;

struct Sphere;

impl ProblemBounds for Sphere {
    fn dimensions(&self) -> usize { 2 }
    fn lower_bounds(&self) -> &[f64] { &[-5.0, -5.0] }
    fn upper_bounds(&self) -> &[f64] { &[5.0, 5.0] }
}

impl Problem for Sphere {
    fn evaluate(&mut self, genes: &[f64]) -> ProblemResult<f64> {
        Ok(genes.iter().map(|value| value * value).sum())
    }
}

let problem = Sphere;
let mut ga = RealGa::builder(problem)
    .population_size(20)
    .stop_condition(
        StopCondition::target_fitness_below(1e-3).or(StopCondition::max_generations(250)),
    )
    .build()
    .unwrap();
let mut rng = rand::rngs::StdRng::seed_from_u64(7);
let report = ga.run(&mut rng).unwrap();
println!("best: {:?} => {}", report.best_solution, report.best_fitness);
```

### Experiment analytics

Each `RealGaReport` and `Nsga2Report` now exposes an `experiment` payload that
captures the final population, the best individual, and [`RunStats`](https://docs.rs/jeans/latest/jeans/struct.RunStats.html) time-series
metrics such as the best/mean/median fitness plus a diversity score for every
generation. These analytics simplify downstream visualization or checkpointing
without requiring ad-hoc bookkeeping inside your optimization loop.

The `experiment` field also records [`ExperimentMetadata`](https://docs.rs/jeans/latest/jeans/struct.ExperimentMetadata.html), including the number
of generations that ran and a description of the RNG that was supplied to the
engine. When you need to persist the payload, enable the optional `serde`
feature:

```toml
jeans = { version = "0.1.4", features = ["serde"] }
```

```rust
let report = ga.run(&mut rng)?;
serde_json::to_string(&report.experiment)?;
```

The [`jeans::ops`](https://docs.rs/jeans/latest/jeans/ops/index.html) module
ships several ready-to-use operators, including SBX and BLX-α crossover plus
polynomial and Gaussian mutation. They can be plugged into the builder when you
need different exploration dynamics:

```rust
use jeans::ops::{BlendAlphaCrossover, GaussianMutation};

let crossover = BlendAlphaCrossover::new(0.3)?;
let mutation = GaussianMutation::new(
    problem.lower_bounds().to_vec(),
    problem.upper_bounds().to_vec(),
    0.15,
    0.2,
)?;
let mut ga = RealGa::builder(problem)
    .crossover(crossover)
    .mutation(mutation)
    .build()?;
```

## Multi-objective optimization

[`Nsga2`](https://docs.rs/jeans/latest/jeans/struct.Nsga2.html) implements the
classic NSGA-II algorithm for problems that return multiple objectives. Define a
[`MultiObjectiveProblem`](https://docs.rs/jeans/latest/jeans/ops/trait.MultiObjectiveProblem.html)
that exposes the bounds, number of objectives, and the evaluation routine, then
configure the engine similarly to [`RealGa`]:

```rust
use jeans::{MultiObjectiveProblem, Nsga2};
use jeans::ops::{ProblemBounds, ProblemResult};
use rand::SeedableRng;

struct LinearFront;

impl ProblemBounds for LinearFront {
    fn dimensions(&self) -> usize { 1 }
    fn lower_bounds(&self) -> &[f64] { &[0.0] }
    fn upper_bounds(&self) -> &[f64] { &[1.0] }
}

impl MultiObjectiveProblem for LinearFront {
    fn objectives(&self) -> usize { 2 }

    fn evaluate(&mut self, genes: &[f64]) -> ProblemResult<Vec<f64>> {
        Ok(vec![genes[0], 1.0 - genes[0]])
    }
}

let problem = LinearFront;
let mut engine = Nsga2::builder(problem)
    .population_size(20)
    .generations(25)
    .build()?;
let mut rng = rand::rngs::StdRng::seed_from_u64(42);
let report = engine.run(&mut rng)?;
println!("found {} non-dominated solutions", report.pareto_solutions.len());
```
