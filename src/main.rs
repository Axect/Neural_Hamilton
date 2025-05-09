use dialoguer::{theme::ColorfulTheme, Select};
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressIterator};
use peroxide::fuga::*;
use rayon::prelude::*;
use rugfield::{grf_with_rng, Kernel};

const V0: f64 = 2f64;
const L: f64 = 1f64;
const NSENSORS: usize = 100;
const BOUNDARY: f64 = 0f64;
const TSTEP: f64 = 1e-3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let normal_or_more_or_much = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Normal or More or Much or Test?")
        .items(&["Normal", "More", "Much", "Test"])
        .default(0)
        .interact()?;
    let n = match normal_or_more_or_much {
        0 => 10000,
        1 => 100000,
        2 => 1000000,
        3 => 4000,
        _ => unreachable!(),
    };
    let folder = match normal_or_more_or_much {
        0 => "data_normal",
        1 => "data_more",
        2 => "data_much",
        3 => "data_test",
        _ => unreachable!(),
    };

    if normal_or_more_or_much == 3 {
        println!("\nGenerate test data...");
        let ds_test = Dataset::generate(n, 789)?;
        let ds_test = ds_test.take(n);
        println!("Take test data: {}", ds_test.data.len());
        let (q_max, p_max) = ds_test.max();
        println!("Max of q: {:.4}, p: {:.4}", q_max, p_max);

        println!("\nWrite data...");
        ds_test.write_parquet(&format!("{}/test.parquet", folder))?;
    } else {
        let train_ratio = 0.8;
        let n_train = (n as f64 * train_ratio).round() as usize;
        let n_val = n - n_train;

        println!("\nGenerate training data...");
        let ds_train = Dataset::generate(n_train, 123)?;
        let ds_train = ds_train.take(n_train);
        println!("Take training data: {}", ds_train.data.len());
        let (q_max, p_max) = ds_train.max();
        println!("Max of q: {:.4}, p: {:.4}", q_max, p_max);

        println!("\nGenerate validation data...");
        let ds_val = Dataset::generate(n_val, 456)?;
        let ds_val = ds_val.take(n_val);
        println!("Take validation data: {}", ds_val.data.len());
        let (q_max, p_max) = ds_val.max();
        println!("Max of q: {:.4}, p: {:.4}", q_max, p_max);

        println!("\nWrite data...");
        ds_train.write_parquet(&format!("{}/train.parquet", folder))?;
        ds_val.write_parquet(&format!("{}/val.parquet", folder))?;
    }

    Ok(())
}

#[derive(Debug, Clone)]
pub struct Dataset {
    pub data: Vec<Data>,
}

impl Dataset {
    pub fn new(data: Vec<Data>) -> Self {
        Dataset { data }
    }

    pub fn take(&self, n: usize) -> Self {
        Dataset {
            data: self.data.iter().take(n).cloned().collect(),
        }
    }

    /// Get max of q and p
    pub fn max(&self) -> (f64, f64) {
        let q_max = self
            .data
            .iter()
            .map(|d| d.q.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
            .fold(f64::NEG_INFINITY, f64::max);
        let p_max = self
            .data
            .iter()
            .map(|d| d.p.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
            .fold(f64::NEG_INFINITY, f64::max);
        (q_max, p_max)
    }

    pub fn generate(n: usize, seed: u64) -> anyhow::Result<Self> {
        let potential = BoundedPotential::generate_potential(n, seed);
        potential.generate_data()
    }

    #[allow(non_snake_case)]
    pub fn unzip(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let V = self
            .data
            .iter()
            .map(|d| d.V.clone())
            .flatten()
            .collect::<Vec<_>>();
        let t = self
            .data
            .iter()
            .map(|d| d.t.clone())
            .flatten()
            .collect::<Vec<_>>();
        let q = self
            .data
            .iter()
            .map(|d| d.q.clone())
            .flatten()
            .collect::<Vec<_>>();
        let p = self
            .data
            .iter()
            .map(|d| d.p.clone())
            .flatten()
            .collect::<Vec<_>>();
        (V, t, q, p)
    }

    #[allow(non_snake_case)]
    pub fn write_parquet(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        // if parent directory does not exist, create it
        let parent = std::path::Path::new(path).parent().unwrap();
        if !parent.exists() {
            std::fs::create_dir_all(parent)?;
        }

        let (V, t, q, p) = self.unzip();

        let mut df = DataFrame::new(vec![]);
        df.push("V", Series::new(V));
        df.push("t", Series::new(t));
        df.push("q", Series::new(q));
        df.push("p", Series::new(p));
        df.print();

        df.write_parquet(path, CompressionOptions::Snappy)?;

        Ok(())
    }
}

#[allow(non_snake_case)]
#[derive(Debug, Clone)]
pub struct Data {
    pub V: Vec<f64>,
    pub t: Vec<f64>,
    pub q: Vec<f64>,
    pub p: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct BoundedPotential {
    pub potential_pair: Vec<(Vec<f64>, Vec<f64>)>,
}

impl BoundedPotential {
    /// Generate Potential via GRF + B-Spline
    ///
    /// # Parameters
    /// - `n`: usize - Number of potentials
    /// - `V0`: f64 - Potential max
    /// - `L`: f64 - Length of domain
    /// - `omega`: f64 - Stability parameter
    /// - `m`: usize - Number of sensors
    /// - `u_b`: Uniform<usize> - Number of GRFs
    /// - `u_l`: Uniform<f64> - Kernel length scale
    /// - `n_q`: Normal<f64> - If b = 1, then normal distribution
    /// - `degree`: usize - Degree of B-Spline
    #[allow(non_snake_case)]
    pub fn generate_potential(n: usize, seed: u64) -> Self {
        // For safety
        let n_cand = (n as f64 * 1.5).round() as usize;

        // Declare parameters
        let omega = 0.05;
        let u_b = Uniform(2, 7);
        let u_l = Uniform(0.01, 0.2);
        let degree = 3;
        let mut rng = stdrng_from_seed(seed);

        // Generate GRF
        let b = u_b
            .sample_with_rng(&mut rng, n_cand)
            .into_iter()
            .map(|x| x.ceil() as usize)
            .collect::<Vec<_>>();
        let l = u_l.sample_with_rng(&mut rng, n_cand);

        let grf_vec = b
            .iter()
            .zip(l)
            .progress_with(ProgressBar::new(n_cand as u64))
            .map(|(&b, l)| grf_with_rng(&mut rng, b, Kernel::SquaredExponential(l)))
            .collect::<Vec<_>>();

        // Normalize
        let grf_max_vec = grf_vec.iter().map(|grf| grf.max()).collect::<Vec<_>>();
        let grf_min_vec = grf_vec.iter().map(|grf| grf.min()).collect::<Vec<_>>();
        let grf_max = grf_max_vec.max();
        let grf_min = grf_min_vec.min();
        let mut grf_scaled_vec = grf_vec
            .par_iter()
            .map(|grf| grf.fmap(|x| V0 * (1f64 - 2f64 * (x - grf_min) / (grf_max - grf_min))))
            .collect::<Vec<_>>();

        // Sample nodes
        let q_vec = b
            .iter()
            .map(|&b| {
                let b_step = 1f64 / (b as f64);
                let mut q_sample = vec![0f64];
                for j in 0..b {
                    let omega_1 = if j == 0 { omega } else { omega / 2f64 };
                    let omega_2 = if j == b - 1 { omega } else { omega / 2f64 };
                    let u = Uniform(
                        omega_1 + b_step * (j as f64),
                        b_step * ((j + 1) as f64) - omega_2,
                    );
                    let sampled_normalized_coordinate = u.sample_with_rng(&mut rng, 1)[0];
                    q_sample.push(sampled_normalized_coordinate * L);
                }
                q_sample.push(L);
                q_sample
            })
            .collect::<Vec<_>>();

        // Insert boundary
        grf_scaled_vec.par_iter_mut().for_each(|grf| {
            grf.insert(0, V0);
            grf.push(V0);
        });

        // Cubic B-Spline
        let bounded_potential = q_vec
            .par_iter()
            .zip(grf_scaled_vec.par_iter())
            .progress_with(ProgressBar::new(n_cand as u64))
            .filter_map(|(q, V)| {
                let control_points = q
                    .iter()
                    .zip(V.iter())
                    .map(|(&q, &V)| vec![q, V])
                    .collect::<Vec<_>>();
                let knots = linspace(0, 1, q.len() + 1 - degree);
                match BSpline::clamped(degree, knots, control_points) {
                    Ok(b_spline) => {
                        let t = linspace(0, 1, NSENSORS);
                        let (q_new, potential): (Vec<f64>, Vec<f64>) =
                            b_spline.eval_vec(&t).into_iter().unzip();
                        Some((q_new, potential))
                    }
                    Err(_) => return None,
                }
            })
            .collect::<Vec<_>>();

        BoundedPotential {
            potential_pair: bounded_potential,
        }
    }

    #[allow(non_snake_case)]
    pub fn generate_data(&self) -> anyhow::Result<Dataset> {
        println!("Generate data. Candidates: {}", self.potential_pair.len());

        let data = self
            .potential_pair
            .par_iter()
            .progress_with(ProgressBar::new(self.potential_pair.len() as u64))
            .filter_map(|potential_pair| {
                match solve_hamilton_equation(potential_pair.clone()) {
                    Ok(data) => {
                        let V = &data.V;
                        let q = &data.q;

                        // Filtering
                        if q.iter()
                            .any(|&x| x < -BOUNDARY || x > L + BOUNDARY || !x.is_finite())
                            || V.iter().any(|&x| !x.is_finite() || x.abs() > 10f64)
                        {
                            None
                        } else {
                            Some(data)
                        }
                    }
                    Err(_) => None,
                }
            })
            .collect::<Vec<_>>();
        println!("Generated data: {}", data.len());

        Ok(Dataset::new(data))
    }
}

#[allow(non_snake_case)]
pub struct HamiltonEquation {
    pub V: CubicHermiteSpline,
    pub V_prime: CubicHermiteSpline,
    pub poly_left: Polynomial,
    pub poly_right: Polynomial,
    pub poly_left_grad: Polynomial,
    pub poly_right_grad: Polynomial,
}

impl HamiltonEquation {
    #[allow(non_snake_case)]
    pub fn new(potential_pair: (Vec<f64>, Vec<f64>)) -> anyhow::Result<Self> {
        let (q, V) = potential_pair;
        let cs = cubic_hermite_spline(&q, &V, Quadratic)?;
        let cs_grad = cs.derivative();
        let cs_hess = cs_grad.derivative();
        let V_grad_0 = cs_grad.eval(0f64);
        let V_hess_0 = cs_hess.eval(0f64);
        let V_grad_L = cs_grad.eval(L);
        let V_hess_L = cs_hess.eval(L);

        let poly_left = poly(vec![1f64, 0f64, 0.5 * V_hess_0, V_grad_0, V0]);
        let poly_right_unit = poly(vec![0f64, 0f64, 0f64, 1f64, -L]);
        let poly_right = poly_right_unit.powi(4)
            + poly_right_unit.powi(2) * (0.5 * V_hess_L)
            + poly_right_unit * V_grad_L
            + V0;

        let poly_left_grad = poly_left.derivative();
        let poly_right_grad = poly_right.derivative();

        Ok(HamiltonEquation {
            V: cs,
            V_prime: cs_grad,
            poly_left,
            poly_right,
            poly_left_grad,
            poly_right_grad,
        })
    }

    pub fn eval(&self, q: f64) -> f64 {
        if q < 0f64 {
            self.poly_left.eval(q)
        } else if q > L {
            self.poly_right.eval(q)
        } else {
            self.V.eval(q)
        }
    }

    pub fn eval_vec(&self, q: &[f64]) -> Vec<f64> {
        q.iter()
            .map(|&x| {
                if x < 0f64 {
                    self.poly_left.eval(x)
                } else if x > L {
                    self.poly_right.eval(x)
                } else {
                    self.V.eval(x)
                }
            })
            .collect()
    }

    pub fn eval_grad(&self, q: f64) -> f64 {
        if q < 0f64 {
            self.poly_left_grad.eval(q)
        } else if q > L {
            self.poly_right_grad.eval(q)
        } else {
            self.V_prime.eval(q)
        }
    }
}

impl ODEProblem for HamiltonEquation {
    fn rhs(&self, _t: f64, y: &[f64], dy: &mut [f64]) -> anyhow::Result<()> {
        dy[0] = y[1];
        dy[1] = -self.eval_grad(y[0]);
        Ok(())
    }
}

pub struct YoshidaSolver {
    problem: HamiltonEquation,
}

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

impl YoshidaSolver {
    pub fn new(problem: HamiltonEquation) -> Self {
        YoshidaSolver { problem }
    }

    #[allow(non_snake_case)]
    pub fn solve(
        &self,
        t_span: (f64, f64),
        dt: f64,
        initial_condition: &[f64],
    ) -> anyhow::Result<Data> {
        let hamilton_eq = &self.problem;

        let t_vec = linspace(
            t_span.0,
            t_span.1,
            ((t_span.1 - t_span.0) / dt) as usize + 1,
        );
        let mut q_vec = vec![0f64; t_vec.len()];
        let mut p_vec = vec![0f64; t_vec.len()];
        q_vec[0] = initial_condition[0];
        p_vec[0] = initial_condition[1];

        for i in 1..t_vec.len() {
            let mut q = q_vec[i - 1];
            let mut p = p_vec[i - 1];
            for j in 0..4 {
                q = q + YOSHIDA_COEFF[j] * p * dt;
                p = p + YOSHIDA_COEFF[j + 4] * (-hamilton_eq.eval_grad(q)) * dt;
            }
            q_vec[i] = q;
            p_vec[i] = p;
        }

        let cs_q = cubic_hermite_spline(&t_vec, &q_vec, Quadratic)?;
        let cs_p = cubic_hermite_spline(&t_vec, &p_vec, Quadratic)?;

        let t_vec = linspace(t_span.0, t_span.1, NSENSORS);
        let q_vec = cs_q.eval_vec(&t_vec);
        let p_vec = cs_p.eval_vec(&t_vec);

        let q_uniform = linspace(0, L, NSENSORS);
        let V = hamilton_eq.eval_vec(&q_uniform);
        Ok(Data {
            V,
            t: t_vec,
            q: q_vec,
            p: p_vec,
        })
    }
}

#[allow(non_snake_case)]
pub fn solve_hamilton_equation(potential_pair: (Vec<f64>, Vec<f64>)) -> anyhow::Result<Data> {
    let initial_condition = vec![0f64, 0f64];
    let hamilton_eq = HamiltonEquation::new(potential_pair.clone())?;
    let solver = YoshidaSolver::new(hamilton_eq);
    solver.solve((0f64, 2f64), TSTEP, &initial_condition)
}
