use peroxide::fuga::*;

#[allow(non_snake_case)]
fn main() -> Result<(), Box<dyn Error>> {
    let m = 100;
    let x_vec = linspace(0, 1, m);
    let V_vec = x_vec.fmap(V);

    let t_vec = linspace(0, 2, m);
    let x_vec = t_vec.fmap(position);
    let p_vec = t_vec.fmap(momentum);

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
    let path = format!("{}{}", path, "mff.parquet");
    df.write_parquet(&path, CompressionOptions::Uncompressed)?;

    // ┌─────────────────────────────────────────────────────────┐
    //  RK4
    // └─────────────────────────────────────────────────────────┘
    let problem = MFF;
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
    let path = format!("{}{}", path, "sho_rk4.parquet");
    df.write_parquet(&path, CompressionOptions::Uncompressed)?;

    Ok(())
}

#[allow(non_snake_case)]
fn V(x: f64) -> f64 {
    4f64 * (x - 0.5).abs()
}

#[allow(non_snake_case)]
fn position(t: f64) -> f64 {
    if t < 0.5 {
        2f64 * t.powi(2)
    } else if t < 1.5 {
        -2f64 * t.powi(2) + 4f64 * t - 1f64
    } else {
        2f64 * (2f64 - t).powi(2)
    }
}

#[allow(non_snake_case)]
fn momentum(t: f64) -> f64 {
    if t < 0.5 {
        4f64 * t
    } else if t < 1.5 {
        -4f64 * t + 4f64
    } else {
        -4f64 * (2f64 - t)
    }
}

#[allow(non_snake_case)]
fn dVdx(x: f64) -> f64 {
    4f64 * (x - 0.5).signum()
}

struct MFF;

impl ODEProblem for MFF {
    fn initial_conditions(&self) -> Vec<f64> {
        vec![0f64, 0f64]
    }

    fn rhs(&self, _t: f64, y: &[f64], dy: &mut [f64]) -> anyhow::Result<()> {
        dy[0] = y[1];
        dy[1] = -dVdx(y[0]);
        Ok(())
    }
}
