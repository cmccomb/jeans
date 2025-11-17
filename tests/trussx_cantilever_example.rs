#[path = "../examples/trussx_cantilever/problem.rs"]
mod trussx_problem;

use jeans::ops::Problem;
use trussx_problem::CantileverSizingProblem;

#[test]
fn thicker_members_reduce_penalty() {
    // Arrange
    let mut problem = CantileverSizingProblem::new();
    let thin_candidate = vec![1.2e-4; 3];
    let thick_candidate = vec![5.0e-4; 3];

    // Act
    let thin_fitness = problem
        .evaluate(&thin_candidate)
        .expect("thin candidate should evaluate");
    let thick_fitness = problem
        .evaluate(&thick_candidate)
        .expect("thick candidate should evaluate");
    let thin_deflection = problem
        .analyze_with_deflection(&thin_candidate)
        .expect("thin analysis should succeed")
        .1;
    let thick_deflection = problem
        .analyze_with_deflection(&thick_candidate)
        .expect("thick analysis should succeed")
        .1;

    // Assert
    assert!(
        thick_fitness < thin_fitness,
        "stiffer members should reduce penalties"
    );
    assert!(
        thick_deflection < thin_deflection,
        "stiffer members should reduce tip deflection"
    );
}
