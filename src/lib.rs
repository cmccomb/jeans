#![warn(missing_docs)]
#![warn(missing_doc_code_examples)]

/*! This is a crate for implementing genetic algorithms. Specifically, this is for algorithms whose
    solutions can be represented as a vector of floating point values.
!*/


use rand::Rng;
use rand::seq::SliceRandom;

/// This module provides access to functions commonly used for optimization
pub mod functions {
    /// This function simply sums the elements of the vector
    pub fn summation(x: Vec<f64>) -> f64 {
        x.iter().sum()
    }

    /// This function sums the square of the elements of the vector
    pub fn sphere(x: Vec<f64>) -> f64 {
        let mut f = 0f64;
        for i in 0..x.len() {
            f += x[i]*x[i];
        }
        f
    }
}

/// This is a settings object for storing all of the settings we might care about for a GA.
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
    ///
    pub elitism_fraction: f64,
    ///
    pub maximize_fitness: bool,
    ///
    pub number_of_dimensions: u32,
    ///
    pub upper_bound: Vec<f64>,
    ///
    pub lower_bound: Vec<f64>,
    /// The fitness function used to evaluate how good solutions are
    pub fitness_function: Box<dyn FnMut(Vec<f64>) -> f64>,
}

/// Default values for all settings
impl Default for Settings {
    fn default() -> Self {
        Self {
            fitness_function: Box::new(functions::summation),
            population_size: 100,
            number_of_generations: 100,
            crossover_probability: 0.8,
            mutation_probability: 0.1,
            verbose: true,
            elitism: true,
            elitism_fraction: 0.2,
            maximize_fitness: true,
            number_of_dimensions: 2,
            upper_bound: vec![ 1.0,  1.0],
            lower_bound: vec![-1.0, -1.0],
        }
    }
}

impl Settings {
    /// This functions allows you to set the fitness function
    pub fn set_fitness_function<F: 'static + FnMut(Vec<f64>) -> f64>(&mut self, f: F) {
        if self.maximize_fitness {
            self.fitness_function = Box::new(f);
        } else {
            self.fitness_function = -1.0*Box::new(f);
        }
    }
}


/// This is an optimizer object. It does the actual heavy lifting.
pub struct Optimizer {
    settings: Settings,
    current_population: Population,
    best_fitness: f64,
    best_representation: Vec<f64>
}

///
impl Optimizer {

    ///
    pub fn new(mut settings: Settings) -> Self {
        Self {
            current_population: Population::new(&mut settings),
            settings: settings,
            best_fitness: -f64::INFINITY,
            best_representation: vec![]
        }
    }

    ///
    pub fn solve(&mut self) {
        for _ in 0..self.settings.number_of_generations {
            self.iterate();
        }
    }

    ///
    pub fn report(&self) {
        println!("{}", self.best_fitness);
    }

    ///
    pub fn iterate(&mut self) {
        // New population
        let mut new_population = Population::empty();

        // Elitism
        if self.settings.elitism {
            let number_of_elites: f64 = self.settings.elitism_fraction * self.settings.population_size as f64;
            for i in (self.settings.population_size as usize - number_of_elites as usize)..self.settings.population_size as usize{
                new_population.individuals.push(self.current_population.individuals[i].clone())
            }
        }

        // Crossover

        // Mutation
        for i in 0..self.settings.population_size as usize {
            let mut rng = rand::thread_rng();
            if rng.gen::<f64>() < self.settings.mutation_probability {
                new_population.individuals.push(self.current_population.individuals[i].mutate())
            }
        }

        //Fill in the rest
        while new_population.individuals.len() < self.settings.population_size as usize {
            new_population.individuals.push(Individual::new(&mut self.settings));
        }

        // Get best
        self.current_population = new_population;
        if self.current_population.get_best() > self.best_fitness {
            self.best_fitness = self.current_population.get_best();
        }

        // Sort
        self.current_population.sort();

        self.report();
    }
}


//////////////////////////////////////////
//// Population
//////////////////////////////////////////
struct Population {
    individuals: Vec<Individual>,
}

impl Population {
    fn new(mut settings: &mut Settings) -> Self {
        let mut pop = vec![];
        for _ in 0..settings.population_size {
            pop.push(Individual::new(&mut settings))
        }
        Self {
            individuals: pop,
        }
    }

    fn empty() -> Self {
        Self {
            individuals: vec![],
        }
    }

    fn get_best(&self) -> f64 {
        let mut best_fitness = -f64::INFINITY;
        for i in 0..(*self).individuals.len() {
            if self.individuals[i].fitness > best_fitness {
                best_fitness = self.individuals[i].fitness;
            }
        }
        best_fitness
    }

    fn get_mean(&self) {

    }

    fn get_std(&self) {

    }

    fn get_random(&mut self) -> Option<&mut Individual> {
        self.individuals.choose_mut(&mut rand::thread_rng())
    }

    fn sort(&mut self) {
        self.individuals.sort_unstable_by(|x, y| x.fitness.partial_cmp(&y.fitness).unwrap());
    }
}


//////////////////////////////////////////
//// Individual
//////////////////////////////////////////
struct Individual {
    representation: Vec<f64>,
    fitness: f64,
}

impl Individual {
    fn new(sets: &mut Settings) -> Self {
        let mut rng = rand::thread_rng();
        let mut v: Vec<f64> = vec![];
        for i in 0..sets.number_of_dimensions as usize {
            v.push(rng.gen_range(sets.lower_bound[i], sets.upper_bound[i]));
        }
        Self {
            representation: v.clone(),
            fitness: (sets.fitness_function)(v.clone())
        }
    }

    fn clone(&self) -> Self {
        Self {
            representation: self.representation.clone(),
            fitness: self.fitness,
        }
    }

    // TODO: Make this actual mutation
    fn mutate(&self) -> Self {
        self.clone()
    }
}