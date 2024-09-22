use peroxide::fuga::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let test_data = "data_normal/test.parquet";

    // Load validation data
    let df = DataFrame::read_parquet(test_data)?;
    df.print();
    let val_u: Vec<f64> = df["val_u"].to_vec();

    // RK4
    let mut t_total = vec![];
    let mut x_total = vec![];
    let mut p_total = vec![];
    for i in 0 .. 10 {
        let u = val_u[i * 100 .. (i + 1) * 100].to_vec();
        let ode = PotentialODE::new(&u)?;
        let integrator = RK4;
        let solver = BasicODESolver::new(integrator);
        let (t_vec, xp_vec) = solver.solve(&ode, (0f64, 2f64), 1e-4)?;
        let (x_vec, p_vec): (Vec<f64>, Vec<f64>) = xp_vec.iter().map(|v| (v[0], v[1])).unzip();

        let cs_x = cubic_hermite_spline(&t_vec, &x_vec, Quadratic)?;
        let cs_p = cubic_hermite_spline(&t_vec, &p_vec, Quadratic)?;

        let t_vec = linspace(0, 2, 100);
        let x_vec = cs_x.eval_vec(&t_vec);
        let p_vec = cs_p.eval_vec(&t_vec);

        t_total.extend(t_vec);
        x_total.extend(x_vec);
        p_total.extend(p_vec);
    }

    let mut dg = DataFrame::new(vec![]);
    dg.push("t", Series::new(t_total));
    dg.push("x", Series::new(x_total));
    dg.push("p", Series::new(p_total));
    dg.write_parquet("data_analyze/rk4.parquet", CompressionOptions::Uncompressed)?;
    dg.print();

    Ok(())
}

#[allow(dead_code)]
pub struct PotentialODE {
    cs: CubicHermiteSpline,
    cs_deriv: CubicHermiteSpline,
}

impl PotentialODE {
    pub fn new(potential: &[f64]) -> anyhow::Result<Self> {
        let x = linspace(0, 1, potential.len());
        let cs = cubic_hermite_spline(&x, potential, Quadratic)?;
        let cs_deriv = cs.derivative();

        Ok(Self { cs, cs_deriv })
    }
}

impl ODEProblem for PotentialODE {
    fn initial_conditions(&self) -> Vec<f64> {
        vec![0.0, 0.0]
    }

    fn rhs(&self, _: f64, y: &[f64], dy: &mut [f64]) -> anyhow::Result<()> {
        dy[0] = y[1];
        dy[1] = -self.cs_deriv.eval(y[0]);
        Ok(())
    }
}
