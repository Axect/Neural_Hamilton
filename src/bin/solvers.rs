use indicatif::ParallelProgressIterator;
use peroxide::fuga::*;
use rayon::prelude::*;

const TSTEP: f64 = 2e-2;

macro_rules! solve_hamilton_eq {
    ($integrator:expr, $df:expr) => {{
        let test_V: Vec<f64> = $df["V"].to_vec();
        let test_q: Vec<f64> = $df["q"].to_vec();
        let test_p: Vec<f64> = $df["p"].to_vec();
        let test_t: Vec<f64> = $df["t"].to_vec();
        let samples = test_V.len() / 100;

        let results = (0..samples)
            .into_par_iter()
            .progress_count(samples as u64)
            .map(|i| -> Result<_, anyhow::Error> {
                let V = test_V[i * 100..(i + 1) * 100].to_vec();
                let q_true = test_q[i * 100..(i + 1) * 100].to_vec();
                let p_true = test_p[i * 100..(i + 1) * 100].to_vec();
                // Use actual initial condition from data (not origin)
                // Data is extracted from random time windows, so (q[0], p[0]) != (0, 0)
                let initial_condition = vec![q_true[0], p_true[0]];
                let ode = PotentialODE::new(&V)?;
                let integrator = $integrator;
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
        let mut q_total = vec![];
        let mut p_total = vec![];
        let mut loss_q_total = vec![];
        let mut loss_p_total = vec![];
        let mut loss_total = vec![];
        for (_, q_vec, p_vec, loss_q, loss_p, loss) in results {
            q_total.extend(q_vec);
            p_total.extend(p_vec);
            loss_q_total.extend(loss_q);
            loss_p_total.extend(loss_p);
            loss_total.extend(loss);
        }

        (q_total, p_total, loss_q_total, loss_p_total, loss_total)
    }}
}

#[allow(non_snake_case)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let test_data = "data_test/test.parquet";

    // Load validation data
    let df = DataFrame::read_parquet(test_data)?;
    let V: Vec<f64> = df["V"].to_vec();
    let q: Vec<f64> = df["q"].to_vec();
    let p: Vec<f64> = df["p"].to_vec();
    let t: Vec<f64> = df["t"].to_vec();
    df.print();

    // Solve ODEs using different integrators
    let (q_y4, p_y4, loss_q_y4, loss_p_y4, loss_y4) = solve_hamilton_eq!(YoshidaSolver, &df);
    let (q_rk4, p_rk4, loss_q_rk4, loss_p_rk4, loss_rk4) = solve_hamilton_eq!(RK4, &df);
    let gl4 = GL4 {
        solver: ImplicitSolver::Broyden,
        tol: 1e-6,
        max_step_iter: 100,
    };
    let (q_gl4, p_gl4, loss_q_gl4, loss_p_gl4, loss_gl4) = solve_hamilton_eq!(gl4, &df);

    let mut dh = DataFrame::new(vec![]);
    dh.push("V", Series::new(V));
    dh.push("t", Series::new(t));
    dh.push("q", Series::new(q));
    dh.push("p", Series::new(p));
    dh.push("q_y4", Series::new(q_y4));
    dh.push("p_y4", Series::new(p_y4));
    dh.push("loss_q_y4", Series::new(loss_q_y4));
    dh.push("loss_p_y4", Series::new(loss_p_y4));
    dh.push("loss_y4", Series::new(loss_y4));
    dh.push("q_rk4", Series::new(q_rk4));
    dh.push("p_rk4", Series::new(p_rk4));
    dh.push("loss_q_rk4", Series::new(loss_q_rk4));
    dh.push("loss_p_rk4", Series::new(loss_p_rk4));
    dh.push("loss_rk4", Series::new(loss_rk4));
    dh.push("q_gl4", Series::new(q_gl4));
    dh.push("p_gl4", Series::new(p_gl4));
    dh.push("loss_q_gl4", Series::new(loss_q_gl4));
    dh.push("loss_p_gl4", Series::new(loss_p_gl4));
    dh.push("loss_gl4", Series::new(loss_gl4));

    dh.print();

    dh.write_parquet("data_analyze/test_solvers.parquet", SNAPPY)?;

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

pub struct YoshidaSolver;

const W0: f64 = -1.7024143839193153;
const W1: f64 = 1.3512071919596578;

const YOSHIDA_COEFF: [f64; 8] = [
    W1 / 2f64,
    (W0 + W1) / 2f64,
    (W0 + W1) / 2f64,
    W1 / 2f64,
    W1,
    W0,
    W1,
    0f64,
];

impl ODEIntegrator for YoshidaSolver {
    fn step<P: ODEProblem>(&self, problem: &P, _t: f64, y: &mut [f64], dt: f64) -> anyhow::Result<f64> {
        let mut dy = vec![0f64; y.len()];
        let mut q = y[0];
        let mut p = y[1];
        for j in 0 .. 4 {
            problem.rhs(0f64, &[q, p], &mut dy)?;
            q = q + YOSHIDA_COEFF[j] * dy[0] * dt;
            problem.rhs(0f64, &[q, p], &mut dy)?;
            p = p + YOSHIDA_COEFF[j + 4] * dy[1] * dt;
        }
        y[0] = q;
        y[1] = p;
        Ok(dt)
    }
}
