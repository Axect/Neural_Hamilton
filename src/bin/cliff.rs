use peroxide::fuga::*;

#[allow(non_snake_case)]
fn main() -> Result<(), Box<dyn Error>> {
    let m = 100;
    let x_vec = linspace(0, 1, m);
    let V_vec = x_vec.fmap(V);

    let problem = Cliff;
    //let integrator = DP45 {
    //    tol: 1e-6,
    //    safety_factor: 0.9,
    //    min_step_size: 1e-9,
    //    max_step_size: 1e-2,
    //    max_step_iter: 100,
    //};
    let integrator = GL4 {
        solver: ImplicitSolver::FixedPoint,
        tol: 1e-6,
        max_step_iter: 100,
    };
    let solver = BasicODESolver::new(integrator);
    let (t_vec, xp_vec) = solver.solve(&problem, (0f64, 2f64), 1e-3)?;
    let (x_vec, p_vec): (Vec<f64>, Vec<f64>) = xp_vec.into_iter().map(|xp| (xp[0], xp[1])).unzip();

    let cs_x = cubic_hermite_spline(&t_vec, &x_vec, Quadratic)?;
    let cs_p = cubic_hermite_spline(&t_vec, &p_vec, Quadratic)?;
    let t_vec = linspace(0, 2, m);
    let x_vec = cs_x.eval_vec(&t_vec);
    let p_vec = cs_p.eval_vec(&t_vec);

    let mut df = DataFrame::new(vec![]);
    df.push("u", Series::new(V_vec.clone()));
    df.push("y", Series::new(t_vec));
    df.push("Guy_x", Series::new(x_vec));
    df.push("Guy_p", Series::new(p_vec));
    df.print();

    let path = "data_analyze/";
    if !std::path::Path::new(path).exists() {
        std::fs::create_dir("data_analyze")?;
    }
    let path = format!("{}cliff.parquet", path);
    df.write_parquet(&path, CompressionOptions::Uncompressed)?;

    // ┌─────────────────────────────────────────────────────────┐
    //  RK4
    // └─────────────────────────────────────────────────────────┘
    let problem = Cliff;
    let integrator = RK4;
    let solver = BasicODESolver::new(integrator);
    let (t_vec, xp_vec) = solver.solve(&problem, (0f64, 2f64), 1e-3)?;
    let (x_vec, p_vec): (Vec<f64>, Vec<f64>) = xp_vec.into_iter().map(|xp| (xp[0], xp[1])).unzip();

    let cs_x = cubic_hermite_spline(&t_vec, &x_vec, Quadratic)?;
    let cs_p = cubic_hermite_spline(&t_vec, &p_vec, Quadratic)?;
    let t_vec = linspace(0, 2, m);
    let x_vec = cs_x.eval_vec(&t_vec);
    let p_vec = cs_p.eval_vec(&t_vec);

    let mut df = DataFrame::new(vec![]);
    df.push("u", Series::new(V_vec));
    df.push("y", Series::new(t_vec));
    df.push("Guy_x", Series::new(x_vec));
    df.push("Guy_p", Series::new(p_vec));
    df.print();

    let path = "data_analyze/";
    if !std::path::Path::new(path).exists() {
        std::fs::create_dir("data_analyze")?;
    }
    let path = format!("{}{}", path, "cliff_rk4.parquet");
    df.write_parquet(&path, CompressionOptions::Uncompressed)?;

    Ok(())
}

#[allow(non_snake_case)]
fn V(x: f64) -> f64 {
    //if x < 1f64 {
    //    - 2f64 * (x - 1f64)
    //} else {
    //    2f64
    //}
    let k = 25f64;
    -2f64 * (x - 1f64) * (1f64 - (k * (x - 1f64)).tanh()) / 2f64 + 2f64 * (1f64 + (k * (x - 1f64)).tanh())
}

#[allow(non_snake_case)]
fn dVdx(x: f64) -> f64 {
    //if x < 1f64 {
    //    -2f64
    //} else {
    //    1000f64
    //}
    let k = 25f64;
    (k * x - (k * (x - 1f64)).cosh().powi(2) + (k * (x - 1f64)).cosh() * (k * (x - 1f64)).sinh() + k) / (k * (x - 1f64)).cosh().powi(2)
}

struct Cliff;

impl ODEProblem for Cliff {
    fn initial_conditions(&self) -> Vec<f64> {
        vec![0f64, 0f64]
    }

    fn rhs(&self, _t: f64, y: &[f64], dy: &mut [f64]) -> anyhow::Result<()> {
        dy[0] = y[1];
        dy[1] = -dVdx(y[0]);
        Ok(())
    }
}
