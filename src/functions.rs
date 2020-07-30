/// This module provides access to fitness functions commonly used for optimization.
///
/// These make it very easy to start using `jeans`. For example, you can use one of these built-in
/// functions to quickly modify the fitness function used:
/// ```
/// let mut set = jeans::Settings::default();
/// set.set_fitness_function(jeans::functions::sphere);
/// ```
/// Note that [`jeans::Settings::default()`](struct.Settings.html#fields) will use [`jeans::functions::summation`](fn.summation.html) by default.
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