#![allow(unknown_lints)]
#![warn(missing_docs)]
#![warn(rustdoc::missing_doc_code_examples)]

/*! This is a crate for implementing genetic algorithms. Specifically, this is for algorithms whose
    solutions can be represented as a vector of floating point values.
!*/

use rand::Rng;
use std::convert::TryFrom;

/// This imports a [related library](https://crates.io/crates/benchfun).
use benchfun::SingleObjective;
pub mod core;
pub use crate::core::{Chromosome, Gene, Individual, Population};

/// A settings object for storing all of the settings we might care about for a GA.
///
/// You should usually instantiate this using the default method. All member variables of this struct
/// are public, which makes editing this easy. In most cases, simply reassign the member variables:
/// ```
/// let mut set = jeans::Settings::default();
/// set.elitism = false;
/// ```
/// The most notable exception is for the `fitness_function` field. To set that, you have to use a
/// special method:
/// ```
/// let mut set = jeans::Settings::default();
/// set.set_fitness_function(| x: Vec<f64> | x[0] + x[1])
/// ```
pub struct Settings {
    /// The size of the population
    pub population_size: u32,
    /// The number of generations
    pub number_of_generations: u32,
    /// The crossover probability
    pub crossover_probability: f64,
    /// The probability of mutation
    pub mutation_probability: f64,
    /// A `bool` indicating whether or not code should run in verbose mode
    pub verbose: bool,
    /// A `bool` indicating whether or not to implement elitism
    pub elitism: bool,
    /// The fraction of each generation that should be carried forward via elitism
    pub elitism_fraction: f64,
    /// Whether the package should maximize fitness (`true`) or minimize fitness (`false`)
    pub maximize_fitness: bool,
    /// The number of dimensions of the problem space
    pub number_of_dimensions: u32,
    /// The upper bounds for each dimension of the problem space
    pub upper_bound: Vec<f64>,
    /// The lower bounds for each dimension of the problem space
    pub lower_bound: Vec<f64>,
    /// The fitness function used to evaluate how good solutions are
    pub fitness_function: Box<dyn FnMut(Vec<f64>) -> f64>,
}

/// Default values for all settings
impl Default for Settings {
    fn default() -> Self {
        Self {
            fitness_function: Box::new(benchfun::single::Sphere::f),
            population_size: 100,
            number_of_generations: 100,
            crossover_probability: 0.5,
            mutation_probability: 0.05,
            verbose: true,
            elitism: true,
            elitism_fraction: 0.1,
            maximize_fitness: true,
            number_of_dimensions: 2,
            upper_bound: vec![1.0, 1.0],
            lower_bound: vec![-1.0, -1.0],
        }
    }
}

impl Settings {
    /// This functions allows you to set the fitness function
    ///
    /// For instance, you can use one of hte built-in function found [here](functions/index.html#functions):
    /// ```
    /// use benchfun::*;
    /// let mut set = jeans::Settings::default();
    /// set.set_fitness_function(benchfun::single::Sphere::f);
    /// ```
    /// Or you can use a lambda to implement your own:
    /// ```
    /// let mut set = jeans::Settings::default();
    /// set.set_fitness_function(| x | x[0] + x[1]);
    /// ```
    /// Or you can define and use a completely new function
    /// ```
    /// fn fitness_whole_pizza(x: Vec<f64>) -> f64 {
    ///     let mut fx = 0f64;
    ///     for i in 0..x.len() {
    ///         fx *= x[i]
    ///     }
    ///     fx
    /// }
    /// let mut set = jeans::Settings::default();
    /// set.set_fitness_function(fitness_whole_pizza);
    /// ```
    pub fn set_fitness_function<F: 'static + FnMut(Vec<f64>) -> f64>(&mut self, f: F) {
        self.fitness_function = Box::new(f);
    }
}

/// This is an optimizer object. It does the actual heavy lifting.
///
/// In order to use the optimizer you first need to create a [`Settings`](struct.Settings.html) struct. That can then be
/// passed to an `Optimizer` struct, and the `solve` method can be called to actually solve the problem.
/// ```
/// let mut set = jeans::Settings::default();
/// let mut opt = jeans::Optimizer::new(set);
/// opt.solve();
/// ```
/// In order to implement and use custom fitness functions, please see [`Settings::set_fitness_function`](struct.Settings.html#impl)
pub struct Optimizer {
    settings: Settings,
    current_population: Population,
    new_population: Population,
    best_fitness: f64,
    best_representation: Vec<f64>,
}

impl Optimizer {
    /// This method enables the creation of a new `Optimizer` struct given a [`Settings`](struct.Settings.html) struct
    #[must_use]
    pub fn new(mut settings: Settings) -> Self {
        Self {
            current_population: Population::new(&mut settings),
            new_population: Population::empty(),
            settings,
            best_fitness: -f64::INFINITY,
            best_representation: vec![],
        }
    }

    /// This method is called to begin the solving process.
    pub fn solve(&mut self) {
        for _ in 0..self.settings.number_of_generations {
            self.iterate();
        }
    }

    /// This method is called to generate a report of the solving process.
    pub fn report(&self) {
        println!(
            "{}, {}",
            self.best_fitness,
            self.current_population.get_mean()
        );
    }

    fn iterate(&mut self) {
        // Elitism
        self.implement_elitism();

        // Crossover
        self.implement_crossover();

        // Mutation
        self.implement_mutation();

        //Fill in the rest
        self.fill_population();

        // Get best
        self.current_population = self.new_population.copy();
        if let Some(best) = self.current_population.best_individual() {
            if best.fitness() > self.best_fitness {
                self.best_fitness = best.fitness();
                self.best_representation = best.chromosome().genes().to_vec();
            }
        }

        // Sort
        self.current_population.sort();
        self.new_population = Population::empty();
        self.report();
    }

    fn implement_elitism(&mut self) {
        // Elitism
        if self.settings.elitism {
            let population_size = self.population_size_usize();
            let number_of_elites =
                self.proportion_to_count(self.settings.elitism_fraction, population_size);
            let start_index = self
                .current_population
                .len()
                .saturating_sub(number_of_elites);
            for elite in &self.current_population.individuals()[start_index..] {
                self.new_population.push(elite.clone());
            }
        }
    }

    fn implement_crossover(&mut self) {
        let population_size = self.population_size_usize();
        let number_of_crosses =
            self.proportion_to_count(self.settings.crossover_probability, population_size);
        let mut rng = rand::thread_rng();
        for _ in 0..number_of_crosses {
            let parent_one = self.current_population.get_random();
            let parent_two = self.current_population.get_random();
            let mut offspring = parent_one.cross(&parent_two, &mut rng);
            let fitness = (self.settings.fitness_function)(offspring.chromosome().genes().to_vec());
            offspring.set_fitness(fitness);
            self.new_population.push(offspring);
        }
    }

    fn implement_mutation(&mut self) {
        let mut rng = rand::thread_rng();
        for individual in self.current_population.individuals() {
            if rng.gen::<f64>() < self.settings.mutation_probability {
                self.new_population.push(individual.mutate());
            }
        }
    }

    fn fill_population(&mut self) {
        let population_size = self.population_size_usize();
        while self.new_population.len() < population_size {
            self.new_population
                .push(Individual::new(&mut self.settings));
        }
    }

    fn population_size_usize(&self) -> usize {
        usize::try_from(self.settings.population_size).expect("population size must fit into usize")
    }

    fn proportion_to_count(&self, proportion: f64, population_size: usize) -> usize {
        let clamped = proportion.clamp(0.0, 1.0);
        let target = clamped * f64::from(self.settings.population_size);
        Self::bounded_float_to_usize(target, population_size)
    }

    fn bounded_float_to_usize(value: f64, max: usize) -> usize {
        if !value.is_finite() {
            return 0;
        }
        let bounded = value.clamp(0.0, Self::usize_to_f64(max));
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        {
            bounded.round() as usize
        }
    }

    fn usize_to_f64(value: usize) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        {
            value as f64
        }
    }
}
