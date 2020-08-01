#![warn(missing_docs)]
#![warn(missing_doc_code_examples)]

/*! This is a crate for implementing genetic algorithms. Specifically, this is for algorithms whose
    solutions can be represented as a vector of floating point values.
!*/

use rand::Rng;
use rand::seq::SliceRandom;

/// This imports a [related library](https://crates.io/crates/benchfun).
use benchfun::*;

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
            upper_bound: vec![ 1.0,  1.0],
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
    best_representation: Vec<f64>
}

impl Optimizer {

    /// This method enables the creation of a new `Optimizer` struct given a [`Settings`](struct.Settings.html) struct
    pub fn new(mut settings: Settings) -> Self {
        Self {
            current_population: Population::new(&mut settings),
            new_population: Population::empty(),
            settings: settings,
            best_fitness: -f64::INFINITY,
            best_representation: vec![]
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
        println!("{}, {}", self.best_fitness, self.current_population.get_mean());
    }

    fn check_bounds(&self, x: Individual) -> bool {
        let mut state = true;
        for i in 0..self.settings.number_of_dimensions as usize{
            if x.representation[i] < self.settings.lower_bound[i] ||
                x.representation[i] > self.settings.upper_bound[i] {
                state = false;
            }
        }
        state
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
        if self.current_population.get_best() > self.best_fitness {
            self.best_fitness = self.current_population.get_best();
        }

        // Sort
        self.current_population.sort();
        self.new_population = Population::empty();
        self.report();
    }

    fn implement_elitism(&mut self) {
        // Elitism
        if self.settings.elitism {
            let number_of_elites: f64 = self.settings.elitism_fraction * self.settings.population_size as f64;
            for i in (self.settings.population_size as usize - number_of_elites as usize)..self.settings.population_size as usize{
                self.new_population.individuals.push(self.current_population.individuals[i].clone())
            }
        }
    }

    fn implement_crossover(&mut self) {
        let number_of_crosses: f64 = self.settings.crossover_probability * self.settings.population_size as f64;
        for _ in (self.settings.population_size as usize - number_of_crosses as usize)..self.settings.population_size as usize{
            let thingone = self.current_population.get_random();
            let thingtwo = self.current_population.get_random();
            let mut newthing = thingone.cross(thingtwo);
            newthing.fitness = (self.settings.fitness_function)(newthing.clone().representation);
            self.new_population.individuals.push(newthing);
        }
    }

    fn implement_mutation(&mut self) {
        for i in 0..self.settings.population_size as usize {
            let mut rng = rand::thread_rng();
            if rng.gen::<f64>() < self.settings.mutation_probability {
                self.new_population.individuals.push(self.current_population.individuals[i].mutate())
            }
        }
    }

    fn fill_population(&mut self) {
        while self.new_population.individuals.len() < self.settings.population_size as usize {
            self.new_population.individuals.push(Individual::new(&mut self.settings));
        }
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

    fn copy(&self) -> Self {
        let mut pop = Population::empty();
        for indi in &self.individuals {
            pop.individuals.push((*indi).clone());
        }
        pop
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

    fn get_mean(&self) -> f64 {
        let mut sum_fitness = 0f64;
        for i in 0..(*self).individuals.len() {
            sum_fitness += self.individuals[i].fitness;
        }
        sum_fitness / (self.individuals.len()as f64)

    }

    fn get_std(&self) {

    }

    fn get_random(&mut self) -> Individual {
        let option = self.individuals.choose_mut(&mut rand::thread_rng());
        match option {
            Some(x) => (*x).clone(),
            None => self.individuals[self.individuals.len()-1].clone()
        }

    }

    fn sort(&mut self) {
        self.individuals.sort_unstable_by(|x, y| x.fitness.partial_cmp(&y.fitness).unwrap());
    }
}


#[cfg(test)]
mod population_tests {
    use super::*;

    #[test]
    fn basic_inst() {
        let x = Population::new(&mut Settings::default());
    }

    #[test]
    fn empty_inst() {
        let x = Population::empty();
    }

    #[test]
    fn sort_check() {
        let mut x = Population::new(&mut Settings::default());
        x.sort();
    }

    #[test]
    fn stat_check() {
        let x = Population::new(&mut Settings::default());
        x.get_best();
        x.get_mean();
        x.get_std();
    }
}

/// This structure is for an individual, essentially representative of a single solution
pub struct Individual {
    representation: Vec<f64>,
    fitness: f64,
}

impl Individual {
    /// This creates a new individual
    fn new(sets: &mut crate::Settings) -> Self {
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

    /// This clones the Individual
    fn clone(&self) -> Self {
        Self {
            representation: self.representation.clone(),
            fitness: self.fitness,
        }
    }

    /// This returns a mutated version of the Individual
    fn mutate(&self) -> Self {
        self.clone()
    }

    /// Crossover
    fn cross(&self, other_individual: Self) -> Self {
        let mut v = vec![];
        let mut rng = rand::thread_rng();
        for i in 0..self.representation.len() {
            if rng.gen::<f64>() < 0.5 {
                v.push(self.representation[i])
            } else {
                v.push(other_individual.representation[i])
            }
        }

        Self {
            representation: v,
            fitness: 0.0
        }

    }
}


#[cfg(test)]
mod individual_tests {
    use super::*;

    #[test]
    fn basic_inst() {
        let x = Individual {
            representation: vec![0.0, 0.0],
            fitness: 0.0
        };
    }

    #[test]
    fn settings_inst() {
        let x = Individual::new(&mut crate::Settings::default());
    }

    #[test]
    fn clone_check() {
        let x = Individual::new(&mut crate::Settings::default());
        let y = x.clone();
    }

    #[test]
    fn mutate_check() {
        let x = Individual::new(&mut crate::Settings::default());
        let y = x.mutate();
    }
}
