use indicatif::ParallelProgressIterator;
use peroxide::fuga::*;
use rayon::prelude::*;

const TSTEP: f64 = 2e-2;

#[allow(non_snake_case)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let test_data = "data_test/test.parquet";

    // Load validation data
    let df = DataFrame::read_parquet(test_data)?;
    df.print();
    let test_V: Vec<f64> = df["V"].to_vec();
    let test_q: Vec<f64> = df["q"].to_vec();
    let test_p: Vec<f64> = df["p"].to_vec();
    let test_t: Vec<f64> = df["t"].to_vec();
    let samples = test_V.len() / 100;

    // RK4 (parallel)
    let results = (0..samples)
        .into_par_iter()
        .progress_count(samples as u64)
        .map(|i| -> Result<_, anyhow::Error> {
            let V = test_V[i * 100..(i + 1) * 100].to_vec();
            let q_true = test_q[i * 100..(i + 1) * 100].to_vec();
            let p_true = test_p[i * 100..(i + 1) * 100].to_vec();
            let initial_condition = vec![0f64, 0f64];
            let ode = PotentialODE::new(&V)?;
            let integrator = RK4;
            let solver = BasicODESolver::new(integrator);
            let (t_vec, qp_vec) = solver.solve(&ode, (0f64, 2f64), TSTEP, &initial_condition)?;
            let (q_vec, p_vec): (Vec<f64>, Vec<f64>) = qp_vec.iter().map(|v| (v[0], v[1])).unzip();

            let cs_q: CubicHermiteSpline = cubic_hermite_spline(&t_vec, &q_vec, Quadratic)?;
            let cs_p: CubicHermiteSpline = cubic_hermite_spline(&t_vec, &p_vec, Quadratic)?;

            let t_vec = test_t[i * 100..(i + 1) * 100].to_vec();
            let q_vec = cs_q.eval_vec(&t_vec);
            let p_vec = cs_p.eval_vec(&t_vec);

            let loss_q = zip_with(|x, y| (x - y).powi(2), &q_true, &q_vec);
            let loss_p = zip_with(|x, y| (x - y).powi(2), &p_true, &p_vec);
            let loss = zip_with(|l1, l2| 0.5 * (l1 + l2), &loss_q, &loss_p);

            Ok((t_vec, q_vec, p_vec, loss_q, loss_p, loss))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Collect results
    let mut t_total = vec![];
    let mut q_total = vec![];
    let mut p_total = vec![];
    let mut loss_q_total = vec![];
    let mut loss_p_total = vec![];
    let mut loss_total = vec![];
    for (t_vec, q_vec, p_vec, loss_q, loss_p, loss) in results {
        t_total.extend(t_vec);
        q_total.extend(q_vec);
        p_total.extend(p_vec);
        loss_q_total.extend(loss_q);
        loss_p_total.extend(loss_p);
        loss_total.extend(loss);
    }

    let mut dg = DataFrame::new(vec![]);
    dg.push("V", Series::new(test_V));
    dg.push("t", Series::new(t_total));
    dg.push("q", Series::new(q_total));
    dg.push("p", Series::new(p_total));
    dg.push("loss_q", Series::new(loss_q_total));
    dg.push("loss_p", Series::new(loss_p_total));
    dg.push("loss", Series::new(loss_total));

    dg.write_parquet("data_analyze/rk4.parquet", CompressionOptions::Snappy)?;
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
        let x = linspace(0f64, 1f64, potential.len());
        let cs = cubic_hermite_spline(&x, potential, Quadratic)?;
        let cs_deriv = cs.derivative();

        Ok(Self { cs, cs_deriv })
    }
}

impl ODEProblem for PotentialODE {
    fn rhs(&self, _: f64, y: &[f64], dy: &mut [f64]) -> anyhow::Result<()> {
        dy[0] = y[1];
        dy[1] = -self.cs_deriv.eval(y[0]);
        Ok(())
    }
}
