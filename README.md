![CI](https://github.com/cmccomb/jeans/actions/workflows/ci.yml/badge.svg?branch=master)

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

The builder wires selection, variation, and termination so you stay focused on
your physics model instead of plumbing GA components.

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
