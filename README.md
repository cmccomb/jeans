![CI](https://github.com/cmccomb/jeans/actions/workflows/ci.yml/badge.svg?branch=master)
[![Crates.io](https://img.shields.io/crates/v/jeans.svg)](https://crates.io/crates/jeans)
[![docs.rs](https://img.shields.io/docsrs/jeans/latest?logo=rust)](https://docs.rs/jeans)

# jeans

`jeans` is an opinionated real-coded genetic algorithm (GA) engine for `Vec<f64>`
solutions in expensive engineering simulations. Unlike the general-purpose
frameworks in the Rust ecosystem—such as Radiate, genevo, or genalg—`jeans`
focuses on pragmatic workflows for industrial design teams who need predictable
operators, deterministic ergonomics, and built-in support for asynchronous
evaluations across compute farms.

## Simple `f(&[f64]) -> f64` API

The core `RealGa` builder accepts problems that expose dimensional bounds and a
plain `f(&[f64]) -> f64` evaluation. You can bring your simulation code, wrap it
in a struct implementing `Problem`, and immediately start optimizing:

```rust
use jeans::{RealGa, StopCondition};
use jeans::ops::{Problem, ProblemBounds, ProblemResult};
use rand::SeedableRng;

struct DragReduction;

impl ProblemBounds for DragReduction {
    fn dimensions(&self) -> usize { 8 }
    fn lower_bounds(&self) -> &[f64] { &[-1.0; 8] }
    fn upper_bounds(&self) -> &[f64] { &[1.0; 8] }
}

impl Problem for DragReduction {
    fn evaluate(&mut self, genes: &[f64]) -> ProblemResult<f64> {
        Ok(simulate_drag(genes))
    }
}

let mut ga = RealGa::builder(DragReduction)
    .population_size(64)
    .stop_condition(StopCondition::max_generations(200))
    .build()?;
let mut rng = rand::rngs::StdRng::seed_from_u64(13);
let report = ga.run(&mut rng)?;
println!("best drag coefficient: {}", report.best_fitness);
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
The builder wires selection, variation, and termination so you stay focused on
your physics model instead of plumbing GA components.
### Asynchronous fitness evaluation

Expensive simulations often expose asynchronous APIs. The
[`jeans::r#async`](https://docs.rs/jeans/latest/jeans/r_async/index.html)
module contains [`AsyncBatchEvaluator`], which batches calls to an
[`AsyncProblem`](https://docs.rs/jeans/latest/jeans/ops/trait.AsyncProblem.html)
implementation and evaluates each candidate inside a Tokio runtime. Engines
such as `RealGa` automatically work with asynchronous evaluators because they
only depend on the
[`SingleObjectiveEvaluator`](https://docs.rs/jeans/latest/jeans/r_async/trait.SingleObjectiveEvaluator.html)
trait:

```rust
use async_trait::async_trait;
use jeans::{RealGa, StopCondition};
use jeans::ops::{AsyncProblem, ProblemBounds, ProblemResult};
use jeans::r#async::AsyncBatchEvaluator;
use rand::SeedableRng;

struct DelayedSphere;

impl ProblemBounds for DelayedSphere {
    fn dimensions(&self) -> usize { 2 }
    fn lower_bounds(&self) -> &[f64] { &[-5.0, -5.0] }
    fn upper_bounds(&self) -> &[f64] { &[5.0, 5.0] }
}

#[async_trait]
impl AsyncProblem for DelayedSphere {
    async fn evaluate_async(&self, genes: &[f64]) -> ProblemResult<f64> {
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        Ok(genes.iter().map(|value| value * value).sum())
    }
}

let evaluator = AsyncBatchEvaluator::new(DelayedSphere)?;
let mut ga = RealGa::builder(evaluator)
    .stop_condition(StopCondition::max_generations(10))
    .build()?;
let mut rng = rand::rngs::StdRng::seed_from_u64(7);
let report = ga.run(&mut rng)?;
println!("best async fitness: {}", report.best_fitness);
```

## Multi-objective optimization

## SBX and polynomial mutation by default

`jeans` defaults to simulated binary crossover (SBX) and polynomial mutation—the
same operators proven on noisy engineering landscapes. They respect the provided
bounds and expose tunable parameters when you need different exploration
pressure, but you can be productive without touching a single knob.

## Asynchronous evaluation focus

Expensive simulations rarely finish together. `jeans` treats async execution as a
first-class concern: problems can spawn asynchronous evaluations, stream
intermediate telemetry, and rely on scheduling hooks to saturate GPU clusters or
simulation farms without custom infrastructure glue.

## NSGA-II multi-objective capabilities

For trade studies with conflicting objectives, `jeans` includes a full
[`Nsga2`](https://docs.rs/jeans/latest/jeans/struct.Nsga2.html) implementation.
Define a `MultiObjectiveProblem` that returns a `Vec<f64>` and use the familiar
builder interface to explore Pareto fronts while keeping the same operator
library as the single-objective engine.

## Design Goals

- **Ergonomics** – Builder-style configuration, opinionated defaults, and tight
  integration with `rand` keep the API approachable for engineers who are new to
  genetic algorithms.
- **Asynchronous execution** – Native async hooks minimize idle hardware time
  when evaluating expensive CFD, FEA, or circuit simulations across clusters.
- **Multi-objective support** – NSGA-II and reusable variation operators make it
  straightforward to reason about Pareto-optimal solutions without rewriting
  your evaluation stack.
let problem = LinearFront;
let mut engine = Nsga2::builder(problem)
    .population_size(20)
    .generations(25)
    .build()?;
let mut rng = rand::rngs::StdRng::seed_from_u64(42);
let report = engine.run(&mut rng)?;
println!("found {} non-dominated solutions", report.pareto_solutions.len());
```

## Examples

The crate ships runnable examples that showcase the higher-level engines:

- `cargo run --example simple_ga` optimizes the Sphere function with explicit
  SBX crossover and polynomial mutation parameters.
- `cargo run --example expensive_async` demonstrates building a Tokio-powered
  [`AsyncProblem`](https://docs.rs/jeans/latest/jeans/ops/trait.AsyncProblem.html)
  that simulates an expensive evaluation before reporting fitness.
- `cargo run --example nsga2_example` runs NSGA-II on a two-objective
  cantilever beam design study, reporting the first few Pareto-optimal designs.

## Development

Contributions are welcome! Please format, lint, and test before opening a pull
request:

```bash
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

Running `cargo doc --no-deps` ensures the documentation keeps compiling during
refactors.
