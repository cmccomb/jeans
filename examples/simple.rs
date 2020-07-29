use jeans;

fn main() {

    // Instantiate and define settings
    let mut settings = jeans::Settings::default();
    settings.set_fitness_function(jeans::functions::sphere);
    settings.number_of_generations = 10;
    settings.population_size = 10;
    settings.lower_bound = vec![0.0, 0.0];

    // Create optimizer and solve
    let mut opt = jeans::Optimizer::new(settings);
    opt.solve();
}
