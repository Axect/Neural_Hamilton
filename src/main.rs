use dialoguer::{theme::ColorfulTheme, Select};
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressIterator};
use peroxide::fuga::anyhow::Result;
use peroxide::fuga::*;
use rayon::prelude::*;
use rugfield::{grf_with_rng, Kernel};

#[allow(non_snake_case)]
fn main() -> std::result::Result<(), Box<dyn Error>> {
    let normal_or_more = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Normal or More or Much?")
        .items(&["Normal", "More", "Much"])
        .default(0)
        .interact()?;
    let n = match normal_or_more {
        0 => 10000,
        1 => 100000,
        _ => 1000000,
    };

    println!("Generate dataset...");
    let ds = Dataset::generate(n, 0.8)?;
    ds.write_parquet(normal_or_more)?;
    println!("Generate dataset complete");

    Ok(())
}

// ┌─────────────────────────────────────────────────────────┐
//  Dataset
// └─────────────────────────────────────────────────────────┘
#[allow(non_snake_case)]
#[derive(Clone)]
pub struct Dataset {
    pub train_u: Matrix,
    pub train_y: Matrix,
    pub train_Gu_x: Matrix,
    pub train_Gu_p: Matrix,
    pub val_u: Matrix,
    pub val_y: Matrix,
    pub val_Gu_x: Matrix,
    pub val_Gu_p: Matrix,
}

impl Dataset {
    #[allow(non_snake_case)]
    pub fn generate(n: usize, f_train: f64) -> Result<Self> {
        // For safety
        let n_cand = (n as f64 * 1.25).round() as usize;

        // Generate GRF
        let m = 100; // # sensors
        //let u_b = WeightedUniform::new(
        //    vec![1f64, 2f64, 2f64, 2f64, 2f64],         // weights
        //    vec![0f64, 1f64, 2f64, 3f64, 4f64, 5f64],   // intervals
        //)?;
        let u_b = Uniform(1, 7);
        let u_l = Uniform(0.01, 0.2);
        let u_n = Normal(0.0, 1.0);
        let mut rng = stdrng_from_seed(42);
        let b = u_b
            .sample_with_rng(&mut rng, n_cand)
            .into_iter()
            .map(|x| x.ceil() as usize)
            .collect::<Vec<_>>();
        let l = u_l.sample_with_rng(&mut rng, n_cand);

        let grf_vec = (0..n_cand)
            .zip(b.iter())
            .zip(l)
            .progress_with(ProgressBar::new(n_cand as u64))
            .map(|((_, &b), l)| if b > 1 {
                grf_with_rng(&mut rng, b, Kernel::SquaredExponential(l))
            } else {
                u_n.sample_with_rng(&mut rng, 1)
            })
            .collect::<Vec<_>>();

        // Normalize
        let grf_max_vec = grf_vec.iter().map(|grf| grf.max()).collect::<Vec<_>>();

        let grf_min_vec = grf_vec.iter().map(|grf| grf.min()).collect::<Vec<_>>();

        let grf_max = grf_max_vec.max();
        let grf_min = grf_min_vec.min();

        let grf_scaled_vec = grf_vec
            .iter()
            .map(|grf| grf.fmap(|x| (x - grf_min) / (grf_max - grf_min)))
            .collect::<Vec<_>>();

        let mut x_vec = vec![];
        for &b in b.iter() {
            let b_step = 1f64 / (b as f64);
            let mut x_sample = vec![0f64];
            if b == 1 {
                let epsilon = 0.05;
                let u = Uniform(epsilon, 1f64 - epsilon);
                x_sample.push(u.sample_with_rng(&mut rng, 1)[0]);
            } else {
                for j in 0..b {
                    let epsilon_1 = if j == 0 { 0.05 } else { 0.025 };
                    let epsilon_2 = if j == b - 1 { 0.05 } else { 0.025 };
                    let u = Uniform(
                        epsilon_1 + b_step * (j as f64),
                        b_step * ((j + 1) as f64) - epsilon_2,
                    );
                    x_sample.push(u.sample_with_rng(&mut rng, 1)[0]);
                }
            }
            x_sample.push(1f64);
            x_vec.push(x_sample);
        }

        let potential_vec = grf_scaled_vec
            .into_par_iter()
            .map(|grf| {
                let mut potential = grf.fmap(|x| 2f64 - 4f64 * x);
                potential.insert(0, 2f64);
                potential.push(2f64);
                potential
            })
            .collect::<Vec<_>>();

        type ParReturn = (
            Vec<Vec<f64>>,
            (
                Vec<Vec<f64>>,
                (Vec<Vec<f64>>, (Vec<Vec<f64>>, Vec<Vec<f64>>)),
            ),
        );
        let (x_vec, (potential_vec, (y_vec, (Gu_x_vec, Gu_p_vec)))): ParReturn = potential_vec
            .par_iter()
            .zip(x_vec.par_iter())
            .progress_with(ProgressBar::new(n_cand as u64))
            .filter_map(|(potential, x)| match solve_grf_ode(potential, x) {
                Ok((t_vec, y_vec, p_vec)) => {
                    Some((x.clone(), (potential.clone(), (t_vec, (y_vec, p_vec)))))
                }
                Err(_) => None,
            })
            .unzip();

        // Filter odd data
        let mut ics = vec![];
        for (i, (u, gu)) in potential_vec.iter().zip(Gu_x_vec.iter()).enumerate() {
            if gu.iter().any(|gu| gu.abs() > 1.1 || !gu.is_finite()) || u.iter().any(|u| u.abs() > 10f64 || !u.is_finite()) {
                continue;
            }
            ics.push(i);
        }
        let x_vec = ics.iter().map(|i| x_vec[*i].clone()).collect::<Vec<_>>();
        let potential_vec = ics
            .iter()
            .map(|i| potential_vec[*i].clone())
            .collect::<Vec<_>>();
        let y_vec = ics.iter().map(|i| y_vec[*i].clone()).collect::<Vec<_>>();
        let Gu_x_vec = ics.iter().map(|i| Gu_x_vec[*i].clone()).collect::<Vec<_>>();
        let Gu_p_vec = ics.iter().map(|i| Gu_p_vec[*i].clone()).collect::<Vec<_>>();

        let sensors = linspace(0, 1, m);
        let u_vec = potential_vec
            .par_iter()
            .zip(x_vec.par_iter())
            .map(|(potential, x)| {
                let degree = 3;
                let control_points = x.iter()
                    .zip(potential.iter())
                    .map(|(&x, &y)| vec![x, y])
                    .collect::<Vec<_>>();
                let knots = linspace(0, 1, x.len() + 1 - degree);
                let bspline = BSpline::clamped(degree, knots, control_points)?;
    
                let t = linspace(0, 1, 100);
                let (x_new, potential): (Vec<f64>, Vec<f64>) = bspline.eval_vec(&t).into_iter().unzip();

                let cs = cubic_hermite_spline(&x_new, &potential, Quadratic)?;
                Ok(cs.eval_vec(&sensors))
            })
            .collect::<Result<Vec<_>>>()?;

        let n_train = (n as f64 * f_train).round() as usize;
        let n_val = n - n_train;

        println!("total data: {}", u_vec.len());
        println!("n_train: {}", n_train);
        println!("n_val: {}", n_val);

        let train_u = u_vec.iter().take(n_train).cloned().collect::<Vec<_>>();
        let train_y = y_vec.iter().take(n_train).cloned().collect::<Vec<_>>();
        let train_Gu_x = Gu_x_vec.iter().take(n_train).cloned().collect::<Vec<_>>();
        let train_Gu_p = Gu_p_vec.iter().take(n_train).cloned().collect::<Vec<_>>();

        let val_u = u_vec
            .iter()
            .skip(n_train)
            .take(n_val)
            .cloned()
            .collect::<Vec<_>>();
        let val_y = y_vec
            .iter()
            .skip(n_train)
            .take(n_val)
            .cloned()
            .collect::<Vec<_>>();
        let val_Gu_x = Gu_x_vec
            .iter()
            .skip(n_train)
            .take(n_val)
            .cloned()
            .collect::<Vec<_>>();
        let val_Gu_p = Gu_p_vec
            .iter()
            .skip(n_train)
            .take(n_val)
            .cloned()
            .collect::<Vec<_>>();

        Ok(Self {
            train_u: py_matrix(train_u),
            train_y: py_matrix(train_y),
            train_Gu_x: py_matrix(train_Gu_x),
            train_Gu_p: py_matrix(train_Gu_p),
            val_u: py_matrix(val_u),
            val_y: py_matrix(val_y),
            val_Gu_x: py_matrix(val_Gu_x),
            val_Gu_p: py_matrix(val_Gu_p),
        })
    }

    #[allow(non_snake_case)]
    pub fn train_set(&self) -> (Matrix, Matrix, Matrix, Matrix) {
        (
            self.train_u.clone(),
            self.train_y.clone(),
            self.train_Gu_x.clone(),
            self.train_Gu_p.clone(),
        )
    }

    #[allow(non_snake_case)]
    pub fn val_set(&self) -> (Matrix, Matrix, Matrix, Matrix) {
        (
            self.val_u.clone(),
            self.val_y.clone(),
            self.val_Gu_x.clone(),
            self.val_Gu_p.clone(),
        )
    }

    #[allow(non_snake_case)]
    pub fn write_parquet(
        &self,
        normal_or_more: usize,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let data_folder = match normal_or_more {
            0 => "data_normal",
            1 => "data_more",
            _ => "data_much",
        };
        if !std::path::Path::new(data_folder).exists() {
            std::fs::create_dir(data_folder)?;
        }

        let (train_u, train_y, train_Gu_x, train_Gu_p) = self.train_set();
        let (val_u, val_y, val_Gu_x, val_Gu_p) = self.val_set();

        let mut df = DataFrame::new(vec![]);
        df.push("train_u", Series::new(train_u.data));
        df.push("train_y", Series::new(train_y.data));
        df.push("train_Gu_x", Series::new(train_Gu_x.data));
        df.push("train_Gu_p", Series::new(train_Gu_p.data));

        let train_path = format!("{}/train.parquet", data_folder);
        df.write_parquet(&train_path, CompressionOptions::Uncompressed)?;

        let mut df = DataFrame::new(vec![]);
        df.push("val_u", Series::new(val_u.data));
        df.push("val_y", Series::new(val_y.data));
        df.push("val_Gu_x", Series::new(val_Gu_x.data));
        df.push("val_Gu_p", Series::new(val_Gu_p.data));

        let val_path = format!("{}/val.parquet", data_folder);
        df.write_parquet(&val_path, CompressionOptions::Uncompressed)?;

        Ok(())
    }
}

#[allow(dead_code)]
pub struct GRFODE {
    cs: CubicHermiteSpline,
    cs_deriv: CubicHermiteSpline,
}

impl GRFODE {
    pub fn new(potential: &[f64], x: &[f64]) -> anyhow::Result<Self> {
        let degree = 3;
        let control_points = x.iter()
            .zip(potential.iter())
            .map(|(&x, &y)| vec![x, y])
            .collect::<Vec<_>>();
        let knots = linspace(0, 1, x.len() + 1 - degree);
        let bspline = BSpline::clamped(degree, knots, control_points)?;
    
        let t = linspace(0, 1, 100);
        let (x_new, potential): (Vec<f64>, Vec<f64>) = bspline.eval_vec(&t).into_iter().unzip();

        let cs = cubic_hermite_spline(&x_new, &potential, Quadratic)?;
        let cs_deriv = cs.derivative();
        Ok(Self { cs, cs_deriv })
    }
}

impl ODEProblem for GRFODE {
    fn initial_conditions(&self) -> Vec<f64> {
        vec![0f64, 0f64]
    }

    fn rhs(&self, _t: f64, y: &[f64], dy: &mut [f64]) -> anyhow::Result<()> {
        dy[0] = y[1]; // dot(x) = p
        dy[1] = -self.cs_deriv.eval(y[0]); // dot(p) = - partial V / partial x
        Ok(())
    }
}

pub fn solve_grf_ode(
    potential: &[f64],
    x: &[f64],
) -> anyhow::Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let grf_ode = GRFODE::new(potential, x)?;
    let integrator = GL4 {
        solver: ImplicitSolver::FixedPoint,
        tol: 1e-6,
        max_step_iter: 100,
    };
    let solver = BasicODESolver::new(integrator);
    let (t_vec, xp_vec) = solver.solve(&grf_ode, (0f64, 2f64), 1e-3)?;
    let (x_vec, p_vec): (Vec<f64>, Vec<f64>) = xp_vec.into_iter().map(|xp| (xp[0], xp[1])).unzip();

    // Uniform nodes
    let n = 100;
    let cs_x = cubic_hermite_spline(&t_vec, &x_vec, Quadratic)?;
    let cs_p = cubic_hermite_spline(&t_vec, &p_vec, Quadratic)?;
    let t_vec = linspace(0, 2, n);
    let x_vec = cs_x.eval_vec(&t_vec);
    let p_vec = cs_p.eval_vec(&t_vec);

    Ok((t_vec, x_vec, p_vec))
}

/// Sabitsky-Golay Filter for smoothing (5-point quadratic)
pub fn sabitsky_golay_filter_5(y: &[f64]) -> Vec<f64> {
    let n = y.len();
    let mut y_smooth = vec![0f64; n];
    for i in 0..n {
        if i < 2 || i > n - 3 {
            y_smooth[i] = y[i];
        } else {
            y_smooth[i] = (-3f64 * (y[i - 2] + y[i + 2])
                + 12f64 * (y[i - 1] + y[i + 1])
                + 17f64 * y[i])
                / 35f64;
        }
    }
    y_smooth
}
