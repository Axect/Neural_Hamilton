use indicatif::{ParallelProgressIterator, ProgressBar, ProgressIterator};
use peroxide::fuga::*;
use rayon::prelude::*;
use rugfield::{grf_with_rng, Kernel};

const V0: f64 = 2f64;
const L: f64 = 1f64;
const NSENSORS: usize = 100;
const BOUNDARY: f64 = 0f64;
const TSTEP: f64 = 1e-3;
const NDIFFCONFIG: usize = 2; // Number of different configurations of time per potential
const T_TOTAL: f64 = 4f64; // Total integration time for long trajectory
const T_WINDOW: f64 = 2f64; // Time window size for each sample

// Target data counts (exact output after filtering)
const TARGET_TRAIN: usize = 80_000 * 10;   // 800,000
const TARGET_VAL: usize = 20_000 * 10;     // 200,000
const TARGET_TEST: usize = 10_000 * 10;    // 100,000
const SAFETY_FACTOR: f64 = 2.0;            // Generate 2x candidates to account for filtering (~50% pass rate)

// --- Yoshida 4th Order Coefficients ---
const W0_4TH: f64 = -1.7024143839193153;
const W1_4TH: f64 = 1.3512071919596578;
const YOSHIDA_4TH_COEFF: [f64; 8] = [
    W1_4TH / 2f64,            // c1
    (W0_4TH + W1_4TH) / 2f64, // c2
    (W0_4TH + W1_4TH) / 2f64, // c3
    W1_4TH / 2f64,            // c4
    W1_4TH,                   // d1
    W0_4TH,                   // d2
    W1_4TH,                   // d3
    0f64,                     // d4
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Solver {
    Yoshida4th,
    RK4,
    GL4,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Interactive selection of data generation mode
    //let items = &["Normal", "More", "Much", "Precise", "Test"];
    //let selection = Select::with_theme(&ColorfulTheme::default())
    //    .with_prompt("Select data generation mode:")
    //    .items(items)
    //    .default(0)
    //    .interact()?;

    // Non-interactive selection for testing
    let args: Vec<String> = std::env::args().collect();
    let selection = if args.len() > 1 {
        args[1].parse::<usize>().unwrap_or(0)
    } else {
        println!("No selection provided, defaulting to Normal mode.");
        0 // Default to Normal if no argument is provided
    };

    match selection {
        0 | 1 => {
            // Normal, More (Train/Val with exact target counts)
            let (target_train, target_val, folder, order) = match selection {
                0 => (TARGET_TRAIN, TARGET_VAL, "data_normal", Solver::Yoshida4th),
                1 => (TARGET_TRAIN * 10, TARGET_VAL * 10, "data_more", Solver::Yoshida4th),
                _ => unreachable!(),
            };

            // Generate candidates with safety margin
            let n_train_cand = (target_train as f64 * SAFETY_FACTOR).ceil() as usize;
            let n_val_cand = (target_val as f64 * SAFETY_FACTOR).ceil() as usize;

            println!("\nGenerate training data (Order: {:?})...", order);
            println!("Target: {}, Generating candidates: {}", target_train, n_train_cand);
            let ds_train_gen = Dataset::generate(n_train_cand, 123, order)?;
            let ds_train = ds_train_gen.take(target_train);
            assert!(
                ds_train.data.len() == target_train,
                "Not enough training data generated! Got {}, need {}. Increase SAFETY_FACTOR.",
                ds_train.data.len(),
                target_train
            );
            println!("Training data: {} (exact)", ds_train.data.len());
            if !ds_train.data.is_empty() {
                let (q_max, p_max) = ds_train.max();
                println!("Max of q: {:.4}, p: {:.4}", q_max, p_max);
            }
            ds_train.write_parquet(&format!("{}/train_cand.parquet", folder))?;

            println!("\nGenerate validation data (Order: {:?})...", order);
            println!("Target: {}, Generating candidates: {}", target_val, n_val_cand);
            let ds_val_gen = Dataset::generate(n_val_cand, 456, order)?;
            let ds_val = ds_val_gen.take(target_val);
            assert!(
                ds_val.data.len() == target_val,
                "Not enough validation data generated! Got {}, need {}. Increase SAFETY_FACTOR.",
                ds_val.data.len(),
                target_val
            );
            println!("Validation data: {} (exact)", ds_val.data.len());
            if !ds_val.data.is_empty() {
                let (q_max, p_max) = ds_val.max();
                println!("Max of q: {:.4}, p: {:.4}", q_max, p_max);
            }
            ds_val.write_parquet(&format!("{}/val_cand.parquet", folder))?;
        }
        2 => {
            // Test with exact target count
            let target_test = TARGET_TEST;
            let n_test_cand = (target_test as f64 * SAFETY_FACTOR).ceil() as usize;
            let folder = "data_test";
            let order = Solver::Yoshida4th;
            let seed = 8407;

            println!("\nGenerate test data (Order: {:?})...", order);
            println!("Target: {}, Generating candidates: {}", target_test, n_test_cand);
            let ds_test_gen = Dataset::generate(n_test_cand, seed, order)?;
            let ds_test = ds_test_gen.take(target_test);
            assert!(
                ds_test.data.len() == target_test,
                "Not enough test data generated! Got {}, need {}. Increase SAFETY_FACTOR.",
                ds_test.data.len(),
                target_test
            );
            println!("Test data: {} (exact)", ds_test.data.len());
            if !ds_test.data.is_empty() {
                let (q_max, p_max) = ds_test.max();
                println!("Max of q: {:.4}, p: {:.4}", q_max, p_max);
            }
            ds_test.write_parquet(&format!("{}/test_cand.parquet", folder))?;
        }
        _ => unreachable!(),
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

    pub fn generate(n: usize, seed: u64, order: Solver) -> anyhow::Result<Self> {
        let potential_generator = BoundedPotential::generate_potential(n, seed, order);
        potential_generator.generate_data()
    }

    #[allow(non_snake_case)]
    pub fn unzip(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let V = self
            .data
            .iter()
            .flat_map(|d| d.V.clone())
            .collect::<Vec<_>>();
        let t = self
            .data
            .iter()
            .flat_map(|d| d.t.clone())
            .collect::<Vec<_>>();
        let q = self
            .data
            .iter()
            .flat_map(|d| d.q.clone())
            .collect::<Vec<_>>();
        let p = self
            .data
            .iter()
            .flat_map(|d| d.p.clone())
            .collect::<Vec<_>>();
        (V, t, q, p)
    }

    #[allow(non_snake_case)]
    pub fn write_parquet(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
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
        df.write_parquet(path, SNAPPY)?;
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

/// Long trajectory with spline interpolation for window extraction
#[allow(non_snake_case)]
pub struct LongTrajectory {
    pub V: Vec<f64>,           // Potential values at uniform q grid
    pub cs_q: CubicHermiteSpline,  // Spline for q(t)
    pub cs_p: CubicHermiteSpline,  // Spline for p(t)
    pub t_max: f64,            // Maximum time of trajectory
}

impl LongTrajectory {
    /// Extract a window from the long trajectory
    /// t_start: start time of window (in original time)
    /// t_domain: desired output time points (relative to window, i.e., [0, T_WINDOW])
    #[allow(non_snake_case)]
    pub fn extract_window(&self, t_start: f64, t_domain: &[f64]) -> Data {
        // t_domain is in [0, T_WINDOW], shift to [t_start, t_start + T_WINDOW]
        let t_shifted: Vec<f64> = t_domain.iter().map(|&t| t + t_start).collect();

        let q_vec = self.cs_q.eval_vec(&t_shifted);
        let p_vec = self.cs_p.eval_vec(&t_shifted);

        Data {
            V: self.V.clone(),
            t: t_domain.to_vec(),  // Output normalized time [0, T_WINDOW]
            q: q_vec,
            p: p_vec,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BoundedPotential {
    pub potential_pair: Vec<(Vec<f64>, Vec<f64>)>,
    pub solver: Solver,
    pub t_domain_vec: Option<Vec<Vec<f64>>>,
}

impl BoundedPotential {
    #[allow(non_snake_case)]
    pub fn generate_potential(n: usize, seed: u64, solver: Solver) -> Self {
        let n = n / NDIFFCONFIG; // Divide by number of different configurations

        // Accept order
        let n_cand = (n as f64 * 1.1).round() as usize;
        let omega = 0.05;
        let u_b = Uniform(2, 7);
        let u_l = Uniform(0.01, 0.2);
        let degree = 3;
        let mut rng = stdrng_from_seed(seed);

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

        let grf_max_vec = grf_vec.iter().map(|grf| grf.max()).collect::<Vec<_>>();
        let grf_min_vec = grf_vec.iter().map(|grf| grf.min()).collect::<Vec<_>>();
        let grf_max = grf_max_vec.max();
        let grf_min = grf_min_vec.min();
        let mut grf_scaled_vec = grf_vec
            .par_iter()
            .map(|grf| grf.fmap(|x| V0 * (1f64 - 2f64 * (x - grf_min) / (grf_max - grf_min))))
            .collect::<Vec<_>>();

        let q_vec = b
            .iter()
            .map(|&b_val| {
                let b_step = 1f64 / (b_val as f64);
                let mut q_sample = vec![0f64];
                for j in 0..b_val {
                    let omega_1 = if j == 0 { omega } else { omega / 2f64 };
                    let omega_2 = if j == b_val - 1 { omega } else { omega / 2f64 };
                    let u = Uniform(
                        omega_1 + b_step * (j as f64),
                        b_step * ((j + 1) as f64) - omega_2,
                    );
                    q_sample.push(u.sample_with_rng(&mut rng, 1)[0] * L);
                }
                q_sample.push(L);
                q_sample
            })
            .collect::<Vec<_>>();

        grf_scaled_vec.par_iter_mut().for_each(|grf| {
            grf.insert(0, V0);
            grf.push(V0);
        });

        let bounded_potential_pairs = q_vec
            .par_iter()
            .zip(grf_scaled_vec.par_iter())
            .progress_with(ProgressBar::new(n_cand as u64))
            .filter_map(|(q_coords, V_coords)| {
                let control_points = q_coords
                    .iter()
                    .zip(V_coords.iter())
                    .map(|(&qc, &Vc)| vec![qc, Vc])
                    .collect::<Vec<_>>();
                let knots = linspace(0, 1, q_coords.len() + 1 - degree);
                match BSpline::clamped(degree, knots, control_points) {
                    Ok(b_spline) => {
                        let t_eval = linspace(0, 1, NSENSORS);
                        let (q_new, potential_new): (Vec<f64>, Vec<f64>) =
                            b_spline.eval_vec(&t_eval).into_iter().unzip();
                        Some((q_new, potential_new))
                    }
                    Err(_) => None,
                }
            })
            .collect::<Vec<_>>();

        // No longer duplicate potentials - time shift will handle NDIFFCONFIG
        // t_domain_vec is now generated per window in generate_data

        BoundedPotential {
            potential_pair: bounded_potential_pairs,
            solver: solver,
            t_domain_vec: None,  // Will be generated during window extraction
        }
    }

    #[allow(non_snake_case)]
    pub fn generate_data(&self) -> anyhow::Result<Dataset> {
        println!(
            "Generate data with time shift. Unique potentials: {}, Windows per potential: {}",
            self.potential_pair.len(),
            NDIFFCONFIG
        );

        let q_domain = linspace(0f64, L, NSENSORS);

        // Process each potential: solve long trajectory, extract NDIFFCONFIG windows
        let data_vec: Vec<Data> = self
            .potential_pair
            .par_iter()
            .progress_with(ProgressBar::new(self.potential_pair.len() as u64))
            .filter_map(|potential_pair_item| {
                // Solve long trajectory (0 to T_TOTAL)
                let long_traj = match solve_hamilton_long(potential_pair_item.clone()) {
                    Ok(traj) => traj,
                    Err(_) => return None,
                };

                // Check if trajectory stays bounded over entire T_TOTAL
                let check_points = linspace(0f64, T_TOTAL, (T_TOTAL / TSTEP) as usize);
                let q_check = long_traj.cs_q.eval_vec(&check_points);
                if q_check
                    .iter()
                    .any(|&x| x < -BOUNDARY || x > L + BOUNDARY || !x.is_finite())
                {
                    return None;
                }

                // Check V values
                if long_traj.V.iter().any(|&x| !x.is_finite() || x.abs() > 10f64) {
                    return None;
                }

                // Check energy conservation over entire trajectory
                let V_spline = match cubic_hermite_spline(&q_domain, &long_traj.V, Quadratic) {
                    Ok(s) => s,
                    Err(_) => return None,
                };
                let p_check = long_traj.cs_p.eval_vec(&check_points);
                let E: Vec<f64> = q_check
                    .iter()
                    .zip(p_check.iter())
                    .map(|(&q, &p)| V_spline.eval(q) + p * p / 2f64)
                    .collect();
                let E_max = E.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let E_min = E.iter().cloned().fold(f64::INFINITY, f64::min);
                let E_delta = (E_max - E_min) / (E_max + E_min).max(1e-10);
                if E_delta >= 0.001 {
                    return None;
                }

                // Generate NDIFFCONFIG windows with different start times
                // Use deterministic RNG seeded from potential data for reproducibility
                use std::hash::{Hash, Hasher};
                use std::collections::hash_map::DefaultHasher;

                let mut hasher = DefaultHasher::new();
                for &v in &long_traj.V {
                    v.to_bits().hash(&mut hasher);
                }
                let seed = hasher.finish();
                let mut rng = stdrng_from_seed(seed);

                let uniform_t_start = Uniform(0f64, T_TOTAL - T_WINDOW);
                let uniform_t_sample = Uniform(0f64, T_WINDOW);

                let windows: Vec<Data> = (0..NDIFFCONFIG)
                    .filter_map(|_| {
                        // Random start time for this window
                        let t_start = uniform_t_start.sample_with_rng(&mut rng, 1)[0];

                        // Generate random time points within window [0, T_WINDOW]
                        let mut t_domain = uniform_t_sample.sample_with_rng(&mut rng, NSENSORS - 2);
                        t_domain.insert(0, 0f64);
                        t_domain.push(T_WINDOW);
                        t_domain.sort_by(|a, b| a.partial_cmp(b).unwrap());

                        // Extract window
                        let data = long_traj.extract_window(t_start, &t_domain);

                        // Verify window data is valid
                        if data.q.iter().any(|&x| !x.is_finite())
                            || data.p.iter().any(|&x| !x.is_finite())
                        {
                            return None;
                        }

                        Some(data)
                    })
                    .collect();

                if windows.len() == NDIFFCONFIG {
                    Some(windows)
                } else {
                    None
                }
            })
            .flatten()
            .collect();

        println!("Generated data: {} (from {} unique potentials)", data_vec.len(), data_vec.len() / NDIFFCONFIG);
        Ok(Dataset::new(data_vec))
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
        let (q, V_vals) = potential_pair;
        let cs = cubic_hermite_spline(&q, &V_vals, Quadratic)?;
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

    pub fn eval_vec(&self, q_vec: &[f64]) -> Vec<f64> {
        q_vec.iter().map(|&x| self.eval(x)).collect()
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

impl YoshidaSolver {
    pub fn new(problem: HamiltonEquation) -> Self {
        YoshidaSolver { problem }
    }

    #[allow(non_snake_case)]
    fn integration_step_y4(
        &self,
        q_in: f64,
        p_in: f64,
        dt: f64,
        hamilton_eq: &HamiltonEquation,
    ) -> (f64, f64) {
        let mut q = q_in;
        let mut p = p_in;
        for j in 0..4 {
            q = q + YOSHIDA_4TH_COEFF[j] * p * dt;
            p = p + YOSHIDA_4TH_COEFF[j + 4] * (-hamilton_eq.eval_grad(q)) * dt;
        }
        (q, p)
    }

    #[allow(non_snake_case)]
    pub fn solve(
        &self,
        t_span: (f64, f64),
        dt: f64,
        initial_condition: &[f64],
        t_domain: Option<Vec<f64>>,
    ) -> anyhow::Result<Data> {
        let hamilton_eq = &self.problem;
        let num_intervals = ((t_span.1 - t_span.0) / dt).round() as usize;
        let num_points = num_intervals + 1;
        let t_vec_sim = linspace(t_span.0, t_span.1, num_points);

        let mut q_vec_sim = vec![0f64; t_vec_sim.len()];
        let mut p_vec_sim = vec![0f64; t_vec_sim.len()];
        q_vec_sim[0] = initial_condition[0];
        p_vec_sim[0] = initial_condition[1];

        let integration_dt = dt;

        for i in 1..t_vec_sim.len() {
            let (next_q, next_p) = self.integration_step_y4(
                q_vec_sim[i - 1],
                p_vec_sim[i - 1],
                integration_dt,
                hamilton_eq,
            );
            q_vec_sim[i] = next_q;
            p_vec_sim[i] = next_p;
        }

        let cs_q = cubic_hermite_spline(&t_vec_sim, &q_vec_sim, Quadratic)?;
        let cs_p = cubic_hermite_spline(&t_vec_sim, &p_vec_sim, Quadratic)?;

        let t_vec_out = match t_domain {
            Some(t_domain) => t_domain,
            None => linspace(t_span.0, t_span.1, NSENSORS),
        };
        let q_vec_out = cs_q.eval_vec(&t_vec_out);
        let p_vec_out = cs_p.eval_vec(&t_vec_out);

        let q_uniform_pot = linspace(0, L, NSENSORS);
        let V_out = hamilton_eq.eval_vec(&q_uniform_pot);
        Ok(Data {
            V: V_out,
            t: t_vec_out,
            q: q_vec_out,
            p: p_vec_out,
        })
    }
}

#[allow(non_snake_case)]
pub fn solve_hamilton_equation(
    potential_pair: (Vec<f64>, Vec<f64>),
    method: Solver, // Added order parameter
    t_domain: Option<Vec<f64>>,
) -> anyhow::Result<Data> {
    let initial_condition = vec![0f64, 0f64];
    let hamilton_eq = HamiltonEquation::new(potential_pair.clone())?;
    match method {
        Solver::Yoshida4th => {
            let solver = YoshidaSolver::new(hamilton_eq);
            solver.solve((0f64, 2f64), TSTEP, &initial_condition, t_domain)
        }
        Solver::RK4 => {
            let solver = RK4;
            let ode_solver = BasicODESolver::new(solver);
            let initial_condition = vec![0f64, 0f64];
            let (t_vec, x_vec) =
                ode_solver.solve(&hamilton_eq, (0f64, 2f64), TSTEP, &initial_condition)?;
            let x_mat = py_matrix(x_vec);
            let q_vec = x_mat.col(0);
            let p_vec = x_mat.col(1);

            let cs_q = cubic_hermite_spline(&t_vec, &q_vec, Quadratic)?;
            let cs_p = cubic_hermite_spline(&t_vec, &p_vec, Quadratic)?;
            let t_vec_out = match t_domain {
                Some(t_domain) => t_domain,
                None => linspace(0f64, 2f64, NSENSORS),
            };
            let q_vec_out = cs_q.eval_vec(&t_vec_out);
            let p_vec_out = cs_p.eval_vec(&t_vec_out);

            let q_uniform_pot = linspace(0f64, L, NSENSORS);
            let V_out = hamilton_eq.eval_vec(&q_uniform_pot);
            Ok(Data {
                V: V_out,
                t: t_vec_out,
                q: q_vec_out,
                p: p_vec_out,
            })
        }
        Solver::GL4 => {
            let integrator = GL4 {
                solver: ImplicitSolver::Broyden,
                tol: 1e-6,
                max_step_iter: 100,
            };
            let initial_condition = vec![0f64, 0f64];
            let solver = BasicODESolver::new(integrator);
            let (t_vec, x_vec) =
                solver.solve(&hamilton_eq, (0f64, 2f64), TSTEP, &initial_condition)?;
            let x_mat = py_matrix(x_vec);
            let q_vec = x_mat.col(0);
            let p_vec = x_mat.col(1);
            let cs_q = cubic_hermite_spline(&t_vec, &q_vec, Quadratic)?;
            let cs_p = cubic_hermite_spline(&t_vec, &p_vec, Quadratic)?;
            let t_vec_out = match t_domain {
                Some(t_domain) => t_domain,
                None => linspace(0f64, 2f64, NSENSORS),
            };
            let q_vec_out = cs_q.eval_vec(&t_vec_out);
            let p_vec_out = cs_p.eval_vec(&t_vec_out);
            let q_uniform_pot = linspace(0f64, L, NSENSORS);
            let V_out = hamilton_eq.eval_vec(&q_uniform_pot);
            Ok(Data {
                V: V_out,
                t: t_vec_out,
                q: q_vec_out,
                p: p_vec_out,
            })
        }
    }
}

/// Solve Hamilton equation and return LongTrajectory with splines for window extraction
#[allow(non_snake_case)]
pub fn solve_hamilton_long(
    potential_pair: (Vec<f64>, Vec<f64>),
) -> anyhow::Result<LongTrajectory> {
    let hamilton_eq = HamiltonEquation::new(potential_pair)?;

    // Use GL4 for long trajectory (best energy conservation)
    let integrator = GL4 {
        solver: ImplicitSolver::Broyden,
        tol: 1e-6,
        max_step_iter: 100,
    };
    let initial_condition = vec![0f64, 0f64];
    let solver = BasicODESolver::new(integrator);
    let (t_vec, x_vec) = solver.solve(&hamilton_eq, (0f64, T_TOTAL), TSTEP, &initial_condition)?;

    let x_mat = py_matrix(x_vec);
    let q_vec = x_mat.col(0);
    let p_vec = x_mat.col(1);

    // Create splines for interpolation
    let cs_q = cubic_hermite_spline(&t_vec, &q_vec, Quadratic)?;
    let cs_p = cubic_hermite_spline(&t_vec, &p_vec, Quadratic)?;

    // Get V at uniform q grid
    let q_uniform = linspace(0f64, L, NSENSORS);
    let V_out = hamilton_eq.eval_vec(&q_uniform);

    Ok(LongTrajectory {
        V: V_out,
        cs_q,
        cs_p,
        t_max: T_TOTAL,
    })
}
