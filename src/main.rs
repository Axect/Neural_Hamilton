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
pub enum SolverOrder {
    Yoshida4th,
    Yoshida8th,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let items = &["Normal", "More", "Much", "Precise", "Test"];
    let selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select data generation mode:")
        .items(items)
        .default(0)
        .interact()?;

    match selection {
        0 | 1 | 2 => {
            // Normal, More, Much (all Y4 for Train/Val)
            let (n_total_samples, folder, order) = match selection {
                0 => (10000, "data_normal", SolverOrder::Yoshida4th),
                1 => (100000, "data_more", SolverOrder::Yoshida4th),
                2 => (1000000, "data_much", SolverOrder::Yoshida4th),
                _ => unreachable!(),
            };

            let train_ratio = 0.8;
            let n_train = (n_total_samples as f64 * train_ratio).round() as usize;
            let n_val = n_total_samples - n_train;

            println!("\nGenerate training data (Order: {:?})...", order);
            let ds_train_gen = Dataset::generate(n_train, 123, order)?;
            let ds_train = ds_train_gen.take(n_train);
            println!("Take training data: {}", ds_train.data.len());
            if !ds_train.data.is_empty() {
                let (q_max, p_max) = ds_train.max();
                println!("Max of q: {:.4}, p: {:.4}", q_max, p_max);
            }
            ds_train.write_parquet(&format!("{}/train.parquet", folder))?;

            println!("\nGenerate validation data (Order: {:?})...", order);
            let ds_val_gen = Dataset::generate(n_val, 456, order)?;
            let ds_val = ds_val_gen.take(n_val);
            println!("Take validation data: {}", ds_val.data.len());
            if !ds_val.data.is_empty() {
                let (q_max, p_max) = ds_val.max();
                println!("Max of q: {:.4}, p: {:.4}", q_max, p_max);
            }
            ds_val.write_parquet(&format!("{}/val.parquet", folder))?;
        }
        3 => {
            // Precise (Y8 main, Y4 compare for the *exact same potentials*)
            let n_target_pairs = 4000;
            let n_initial_potentials = (n_target_pairs as f64 * 2.5).round() as usize;
            let folder = "data_precise";
            let seed = 789;

            println!("\n--- Precise Mode (Comparison Generation) ---");

            // 1. Create potential candidates
            let potential_generator = BoundedPotential::generate_potential(
                n_initial_potentials,
                seed,
                SolverOrder::Yoshida8th,
            );

            // 2. Create pairs
            match potential_generator.generate_data_for_comparison(n_target_pairs) {
                Ok((ds_y8_final, ds_y4_final)) => {
                    println!(
                        "Successfully generated {} pairs of comparable datasets.",
                        ds_y8_final.data.len()
                    );

                    // Save Y8
                    if !ds_y8_final.data.is_empty() {
                        let (q_max_y8, p_max_y8) = ds_y8_final.max();
                        println!("\nMax of Y8 q: {:.4}, p: {:.4}", q_max_y8, p_max_y8);
                    }
                    println!(
                        "Write Precise Y8 data ({} samples) to {}/compare.parquet...",
                        ds_y8_final.data.len(),
                        folder
                    );
                    ds_y8_final.write_parquet(&format!("{}/compare.parquet", folder))?;

                    // Save Y4
                    if !ds_y4_final.data.is_empty() {
                        let (q_max_y4, p_max_y4) = ds_y4_final.max();
                        println!("\nMax of Y4 q: {:.4}, p: {:.4}", q_max_y4, p_max_y4);
                    }
                    println!(
                        "Write Comparison Y4 data ({} samples) to {}/test.parquet...",
                        ds_y4_final.data.len(),
                        folder
                    );
                    ds_y4_final.write_parquet(&format!("{}/test.parquet", folder))?;
                }
                Err(e) => {
                    eprintln!("Error generating comparison datasets: {}", e);
                }
            }
        }
        4 => {
            // Test
            let n = 4000;
            let folder = "data_test";
            let order = SolverOrder::Yoshida4th;
            let seed = 789;

            println!("\nGenerate test data (Order: {:?})...", order);
            let ds_test_gen = Dataset::generate(n, seed, order)?;
            let ds_test = ds_test_gen.take(n);
            println!("Take test data: {}", ds_test.data.len());
            if !ds_test.data.is_empty() {
                let (q_max, p_max) = ds_test.max();
                println!("Max of q: {:.4}, p: {:.4}", q_max, p_max);
            }
            ds_test.write_parquet(&format!("{}/test.parquet", folder))?;
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

    pub fn generate(n: usize, seed: u64, order: SolverOrder) -> anyhow::Result<Self> {
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
    pub solver_order: SolverOrder, // Store the solver order
}

impl BoundedPotential {
    #[allow(non_snake_case)]
    pub fn generate_potential(n: usize, seed: u64, order: SolverOrder) -> Self {
        // Accept order
        let n_cand = (n as f64 * 1.5).round() as usize;
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

        BoundedPotential {
            potential_pair: bounded_potential_pairs,
            solver_order: order, // Store the order
        }
    }

    #[allow(non_snake_case)]
    pub fn generate_data(&self) -> anyhow::Result<Dataset> {
        // No order param needed here
        println!(
            "Generate data (Order: {:?}). Candidates: {}",
            self.solver_order,
            self.potential_pair.len()
        );
        let data_vec = self
            .potential_pair
            .par_iter()
            .progress_with(ProgressBar::new(self.potential_pair.len() as u64))
            .filter_map(|potential_pair_item| {
                match solve_hamilton_equation(potential_pair_item.clone(), self.solver_order) {
                    // Use stored order
                    Ok(d) => {
                        if d.q
                            .iter()
                            .any(|&x| x < -BOUNDARY || x > L + BOUNDARY || !x.is_finite())
                            || d.V.iter().any(|&x| !x.is_finite() || x.abs() > 10f64)
                        {
                            None
                        } else {
                            Some(d)
                        }
                    }
                    Err(_) => None,
                }
            })
            .collect::<Vec<_>>();
        println!("Generated data: {}", data_vec.len());
        Ok(Dataset::new(data_vec))
    }

    #[allow(non_snake_case)]
    pub fn generate_data_for_comparison(
        &self,
        n_target_pairs: usize,
    ) -> anyhow::Result<(Dataset, Dataset)> {
        println!(
            "Generating Y8/Y4 comparison data. Initial Potential Candidates: {}. Target successful pairs: {}.",
            self.potential_pair.len(),
            n_target_pairs
        );

        let mut successful_y8_data: Vec<Data> = Vec::with_capacity(n_target_pairs);
        let mut successful_y4_data: Vec<Data> = Vec::with_capacity(n_target_pairs);

        // ProgressBar 설정 (전체 포텐셜 후보 수 기준)
        let pb = ProgressBar::new(self.potential_pair.len() as u64);
        pb.set_message("Processing potentials for comparison");

        for potential_item in self.potential_pair.iter() {
            if successful_y8_data.len() >= n_target_pairs {
                break; // 목표 개수 도달 시 중단
            }

            let data_y8_opt =
                solve_hamilton_equation(potential_item.clone(), SolverOrder::Yoshida8th)
                    .ok()
                    .filter(|d| {
                        !d.q.iter()
                            .any(|&x| x < -BOUNDARY || x > L + BOUNDARY || !x.is_finite())
                            && !d.V.iter().any(|&x| !x.is_finite() || x.abs() > 10f64)
                    });

            let data_y4_opt =
                solve_hamilton_equation(potential_item.clone(), SolverOrder::Yoshida4th)
                    .ok()
                    .filter(|d| {
                        !d.q.iter()
                            .any(|&x| x < -BOUNDARY || x > L + BOUNDARY || !x.is_finite())
                            && !d.V.iter().any(|&x| !x.is_finite() || x.abs() > 10f64)
                    });

            if let (Some(d8), Some(d4)) = (data_y8_opt, data_y4_opt) {
                successful_y8_data.push(d8);
                successful_y4_data.push(d4);
            }
            pb.inc(1); // ProgressBar 진행
        }
        pb.finish_with_message("Comparison processing complete");

        println!(
            "Generated {} pairs of comparable Y8 and Y4 data.",
            successful_y8_data.len()
        );

        Ok((
            Dataset::new(successful_y8_data),
            Dataset::new(successful_y4_data),
        ))
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
    fn integration_step_y8(
        &self,
        q_in: f64,
        p_in: f64,
        dt: f64,
        hamilton_eq: &HamiltonEquation,
    ) -> (f64, f64) {
        // Yoshida 8th-order symmetric composition coefficients (γ1…γ8)
        const GAMMA: [f64; 8] = [
            0.7416703643506129,  // γ1 = γ15
            -0.4091008258000316, // γ2 = γ14
            0.1907547102962384,  // γ3 = γ13
            -0.5738624711160823, // γ4 = γ12
            0.2990641813036559,  // γ5 = γ11
            0.3346249182452982,  // γ6 = γ10
            0.3152930923967666,  // γ7 = γ9
            -0.7968879393529163, // γ8 (center)
        ];

        // 1) 앞(m=8) 단계: Y4(dt * γ1)…Y4(dt * γ8)
        let mut q = q_in;
        let mut p = p_in;
        for &g in &GAMMA {
            let (q_new, p_new) = self.integration_step_y4(q, p, g * dt, hamilton_eq);
            q = q_new;
            p = p_new;
        }

        // 2) 뒤(m−1=7) 단계: 대칭되게 γ7…γ1
        for &g in GAMMA[..7].iter().rev() {
            let (q_new, p_new) = self.integration_step_y4(q, p, g * dt, hamilton_eq);
            q = q_new;
            p = p_new;
        }

        (q, p)
    }

    //#[allow(non_snake_case)]
    //fn integration_step_y8(
    //    &self,
    //    q_in: f64,
    //    p_in: f64,
    //    dt: f64,
    //    hamilton_eq: &HamiltonEquation,
    //) -> (f64, f64) {
    //    let mut q = q_in;
    //    let mut p = p_in;
    //    p = p + D_COEFF_8TH[0] * (-hamilton_eq.eval_grad(q)) * dt;
    //    for j in 0..7 {
    //        q = q + C_COEFF_8TH[j] * p * dt;
    //        p = p + D_COEFF_8TH[j + 1] * (-hamilton_eq.eval_grad(q)) * dt;
    //    }
    //    (q, p)
    //}

    #[allow(non_snake_case)]
    pub fn solve(
        &self,
        t_span: (f64, f64),
        dt: f64,
        initial_condition: &[f64],
        order: SolverOrder, // Added order parameter
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
            let (next_q, next_p) = match order {
                SolverOrder::Yoshida4th => self.integration_step_y4(
                    q_vec_sim[i - 1],
                    p_vec_sim[i - 1],
                    integration_dt,
                    hamilton_eq,
                ),
                SolverOrder::Yoshida8th => self.integration_step_y8(
                    q_vec_sim[i - 1],
                    p_vec_sim[i - 1],
                    integration_dt,
                    hamilton_eq,
                ),
            };
            q_vec_sim[i] = next_q;
            p_vec_sim[i] = next_p;
        }

        let cs_q = cubic_hermite_spline(&t_vec_sim, &q_vec_sim, Quadratic)?;
        let cs_p = cubic_hermite_spline(&t_vec_sim, &p_vec_sim, Quadratic)?;

        let t_vec_out = linspace(t_span.0, t_span.1, NSENSORS);
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
    order: SolverOrder, // Added order parameter
) -> anyhow::Result<Data> {
    let initial_condition = vec![0f64, 0f64];
    let hamilton_eq = HamiltonEquation::new(potential_pair.clone())?;
    let solver = YoshidaSolver::new(hamilton_eq);
    solver.solve((0f64, 2f64), TSTEP, &initial_condition, order)
}
