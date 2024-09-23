use peroxide::fuga::*;
use rayon::prelude::*;
use indicatif::ParallelProgressIterator;

#[allow(non_snake_case)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let test_data = "data_normal/test.parquet";

    // Load validation data
    let df = DataFrame::read_parquet(test_data)?;
    df.print();
    let val_u: Vec<f64> = df["val_u"].to_vec();
    let val_Gu_x: Vec<f64> = df["val_Gu_x"].to_vec();
    let val_Gu_p: Vec<f64> = df["val_Gu_p"].to_vec();
    let samples = val_u.len() / 100;

    // RK4 (parallel)
    let results = (0 .. samples)
        .into_par_iter()
        .progress_count(samples as u64)
        .map(|i| -> Result<_, anyhow::Error> {
            let u = val_u[i * 100 .. (i + 1) * 100].to_vec();
            let x_true = val_Gu_x[i * 100 .. (i + 1) * 100].to_vec();
            let p_true = val_Gu_p[i * 100 .. (i + 1) * 100].to_vec();
            let ode = PotentialODE::new(&u)?;
            let integrator = RK4;
            let solver = BasicODESolver::new(integrator);
            let (t_vec, xp_vec) = solver.solve(&ode, (0f64, 2f64), 1e-4)?;
            let (x_vec, p_vec): (Vec<f64>, Vec<f64>) = xp_vec.iter().map(|v| (v[0], v[1])).unzip();

            let cs_x: CubicHermiteSpline = cubic_hermite_spline(&t_vec, &x_vec, Quadratic)?;
            let cs_p: CubicHermiteSpline = cubic_hermite_spline(&t_vec, &p_vec, Quadratic)?;

            let t_vec = linspace(0, 2, 100);
            let x_vec = cs_x.eval_vec(&t_vec);
            let p_vec = cs_p.eval_vec(&t_vec);

            let loss_x = zip_with(|x, y| (x - y).powi(2), &x_true, &x_vec);
            let loss_p = zip_with(|x, y| (x - y).powi(2), &p_true, &p_vec);
            let loss = zip_with(|l1, l2| 0.5 * (l1 + l2), &loss_x, &loss_p);

            Ok((t_vec, x_vec, p_vec, loss_x, loss_p, loss))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let (t_total, x_total, p_total, loss_x_total, loss_p_total, loss_total): (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) = results.into_iter()
        .map(|(t, x, p, loss_x, loss_p, loss)| (
            t.into_iter(),
            x.into_iter(),
            p.into_iter(),
            loss_x.into_iter(),
            loss_p.into_iter(),
            loss.into_iter()
        ))
        .fold(
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()),
            |(mut t_acc, mut x_acc, mut p_acc, mut loss_x_acc, mut loss_p_acc, mut loss_acc), (t, x, p, loss_x, loss_p, loss)| {
                t_acc.extend(t);
                x_acc.extend(x);
                p_acc.extend(p);
                loss_x_acc.extend(loss_x);
                loss_p_acc.extend(loss_p);
                loss_acc.extend(loss);
                (t_acc, x_acc, p_acc, loss_x_acc, loss_p_acc, loss_acc)
            }
        );

    let mut dg = DataFrame::new(vec![]);
    dg.push("t", Series::new(t_total));
    dg.push("x", Series::new(x_total));
    dg.push("p", Series::new(p_total));
    dg.push("loss_x", Series::new(loss_x_total));
    dg.push("loss_p", Series::new(loss_p_total));
    dg.push("loss", Series::new(loss_total));

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
