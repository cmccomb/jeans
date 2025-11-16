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
