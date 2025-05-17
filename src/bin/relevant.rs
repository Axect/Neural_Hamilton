use dialoguer::Select;
use peroxide::fuga::*;
use std::f64::consts::PI;

const NSENSORS: usize = 100;
const BOUNDARY: f64 = 0f64;
const TSTEP: f64 = 2e-2;

// ┌─────────────────────────────────────────────────────────┐
//  Solver Macros
// └─────────────────────────────────────────────────────────┘
macro_rules! yoshida_solve {
    ($potential:expr, $initial_condition:expr) => {{
        if $potential.is_analytic() {
            let t_span = (0f64, 2f64);
            let (q_vec, p_vec) = $potential.analytic_trajectory(t_span, NSENSORS).unwrap();
            let q_uniform = linspace(-BOUNDARY, 1f64 + BOUNDARY, NSENSORS);
            let V = $potential.V_map(&q_uniform);
            Ok(Data { V, t: linspace(t_span.0, t_span.1, NSENSORS), q: q_vec, p: p_vec })
        } else {
            YoshidaSolver::new($potential).solve((0f64, 2f64), TSTEP, $initial_condition)
        }
    }};
}

macro_rules! ode_solve {
    ($potential:expr, $initial_condition:expr, $integrator:expr) => {{
        let t_span = (0f64, 2f64);
        let dt = TSTEP;
        let integrator = $integrator;
        let solver = BasicODESolver::new(integrator);
        let (t_vec, qp_vec) = solver
            .solve(&$potential, t_span, dt, $initial_condition)
            .unwrap();
        let (q_vec, p_vec): (Vec<f64>, Vec<f64>) = qp_vec.into_iter().map(|x| (x[0], x[1])).unzip();
        let cs_q = cubic_hermite_spline(&t_vec, &q_vec, Quadratic).unwrap();
        let cs_p = cubic_hermite_spline(&t_vec, &p_vec, Quadratic).unwrap();
        let t_vec = linspace(t_span.0, t_span.1, NSENSORS);
        let q_vec = cs_q.eval_vec(&t_vec);
        let p_vec = cs_p.eval_vec(&t_vec);
        (q_vec, p_vec)
    }};
}

// ┌─────────────────────────────────────────────────────────┐
//  Main function
// └─────────────────────────────────────────────────────────┘
#[allow(non_snake_case)]
fn main() -> Result<(), Box<dyn Error>> {
    // Set up potential selection
    let potentials = vec![
        "SHO",
        "DoubleWell",
        "Morse",
        "Pendulum",
        "MirroredFreeFall",
        "SoftenedMirroredFreeFall",
    ];
    let potential = Select::new()
        .with_prompt("Choose potential")
        .items(&potentials)
        .default(0)
        .interact()?;

    // Define initial conditions
    let initial_condition = [0f64, 0f64];

    // Solve using Yoshida method
    let data = match potential {
            0 => yoshida_solve!(SHO, &initial_condition),
            1 => yoshida_solve!(DoubleWell, &initial_condition),
            2 => yoshida_solve!(
                Morse {
                    a: 3f64 * ((1f64 + 5f64.sqrt()) / 2f64).ln(),
                    D_e: 8f64 / (5f64.sqrt() - 1f64).powi(2),
                    r_e: 1f64 / 3f64
                },
                &initial_condition
            ),
            3 => yoshida_solve!(Pendulum { theta_L: PI / 3f64 }, &initial_condition),
            4 => yoshida_solve!(MirroredFreeFall, &initial_condition),
            5 => yoshida_solve!(SoftenedMirroredFreeFall, &initial_condition),
            _ => unreachable!(),
        }?;

    // Solve using RK4 method
    let rk4_data = match potential {
            0 => ode_solve!(SHO, &initial_condition, RK4),
            1 => ode_solve!(DoubleWell, &initial_condition, RK4),
            2 => ode_solve!(
                Morse {
                    a: 3f64 * ((1f64 + 5f64.sqrt()) / 2f64).ln(),
                    D_e: 8f64 / (5f64.sqrt() - 1f64).powi(2),
                    r_e: 1f64 / 3f64
                },
                &initial_condition,
                RK4
            ),
            3 => ode_solve!(Pendulum { theta_L: PI / 3f64 }, &initial_condition, RK4),
            4 => ode_solve!(MirroredFreeFall, &initial_condition, RK4),
            5 => ode_solve!(SoftenedMirroredFreeFall, &initial_condition, RK4),
            _ => unreachable!(),
        };

    // Solve using GL4 method
    let gl4 = GL4 {
        solver: ImplicitSolver::Broyden,
        tol: 1e-6,
        max_step_iter: 100,
    };
    let gl4_data = match potential {
            0 => ode_solve!(SHO, &initial_condition, gl4),
            1 => ode_solve!(DoubleWell, &initial_condition, gl4),
            2 => ode_solve!(
                Morse {
                    a: 3f64 * ((1f64 + 5f64.sqrt()) / 2f64).ln(),
                    D_e: 8f64 / (5f64.sqrt() - 1f64).powi(2),
                    r_e: 1f64 / 3f64
                },
                &initial_condition,
                gl4
            ),
            3 => ode_solve!(Pendulum { theta_L: PI / 3f64 }, &initial_condition, gl4),
            4 => ode_solve!(MirroredFreeFall, &initial_condition, gl4),
            5 => ode_solve!(SoftenedMirroredFreeFall, &initial_condition, gl4),
            _ => unreachable!(),
        };

    // Solve using RKF78 method
    let rkf78 = RKF78 {
        tol: 1e-6,
        safety_factor: 0.9,
        min_step_size: 1e-6,
        max_step_size: TSTEP * 10f64,
        max_step_iter: 100,
    };
    let rkf78_data = match potential {
            0 => ode_solve!(SHO, &initial_condition, rkf78),
            1 => ode_solve!(DoubleWell, &initial_condition, rkf78),
            2 => ode_solve!(
                Morse {
                    a: 3f64 * ((1f64 + 5f64.sqrt()) / 2f64).ln(),
                    D_e: 8f64 / (5f64.sqrt() - 1f64).powi(2),
                    r_e: 1f64 / 3f64
                },
                &initial_condition,
                rkf78
            ),
            3 => ode_solve!(Pendulum { theta_L: PI / 3f64 }, &initial_condition, rkf78),
            4 => ode_solve!(MirroredFreeFall, &initial_condition, rkf78),
            5 => ode_solve!(SoftenedMirroredFreeFall, &initial_condition, rkf78),
            _ => unreachable!(),
        };

    let potentials = vec!["sho", "double_well", "morse", "pendulum", "mff", "smff"];
    let potential_name = potentials[potential];

    let mut df = DataFrame::new(vec![]);

    let V = data.V.clone();
    let t = data.t.clone();
    let q = data.q.clone();
    let p = data.p.clone();
    let q_rk4 = rk4_data.0.clone();
    let p_rk4 = rk4_data.1.clone();
    let q_gl4 = gl4_data.0.clone();
    let p_gl4 = gl4_data.1.clone();
    let q_rkf78 = rkf78_data.0.clone();
    let p_rkf78 = rkf78_data.1.clone();

    df.push("V", Series::new(V));
    df.push("t", Series::new(t));
    df.push("q", Series::new(q));
    df.push("p", Series::new(p));
    df.push("q_rk4", Series::new(q_rk4));
    df.push("p_rk4", Series::new(p_rk4));
    df.push("q_gl4", Series::new(q_gl4));
    df.push("p_gl4", Series::new(p_gl4));
    df.push("q_rkf78", Series::new(q_rkf78));
    df.push("p_rkf78", Series::new(p_rkf78));
    df.print();
    let file_name = format!("data_analyze/{}.parquet", potential_name);
    df.write_parquet(file_name.as_str(), CompressionOptions::Uncompressed)?;

    Ok(())
}

// ┌─────────────────────────────────────────────────────────┐
//  Potential Trait and Implementation
// └─────────────────────────────────────────────────────────┘
pub trait Potential {
    #[allow(non_snake_case)]
    fn V(&self, q: f64) -> f64;
    #[allow(non_snake_case)]
    fn dV(&self, q: f64) -> f64;
    fn is_analytic(&self) -> bool {
        self.analytic_solution(0f64).is_some()
    }
    fn analytic_solution(&self, t: f64) -> Option<(f64, f64)>;
    fn analytic_trajectory(&self, t_span: (f64, f64), m: usize) -> Option<(Vec<f64>, Vec<f64>)> {
        let t_vec = linspace(t_span.0, t_span.1, m);
        let mut q_vec = vec![0f64; t_vec.len()];
        let mut p_vec = vec![0f64; t_vec.len()];
        for i in 0..t_vec.len() {
            let t = t_vec[i];
            match self.analytic_solution(t) {
                Some((q, p)) => {
                    q_vec[i] = q;
                    p_vec[i] = p;
                }
                None => return None,
            }
        }
        Some((q_vec, p_vec))
    }

    #[allow(non_snake_case)]
    fn V_map(&self, q: &Vec<f64>) -> Vec<f64> {
        q.iter().map(|&x| self.V(x)).collect()
    }
}

// ┌─────────────────────────────────────────────────────────┐
//  ODEProblem Implementation Macro
// └─────────────────────────────────────────────────────────┘
macro_rules! impl_ode_problem {
    ($type:ty) => {
        impl ODEProblem for $type {
            fn rhs(&self, _t: f64, y: &[f64], dy: &mut [f64]) -> anyhow::Result<()> {
                dy[0] = y[1];
                dy[1] = -self.dV(y[0]);
                Ok(())
            }
        }
    };
}

// Implement ODEProblem for all potential types
impl_ode_problem!(SHO);
impl_ode_problem!(DoubleWell);
impl_ode_problem!(Morse);
impl_ode_problem!(Pendulum);
impl_ode_problem!(MirroredFreeFall);
impl_ode_problem!(SoftenedMirroredFreeFall);

// ┌─────────────────────────────────────────────────────────┐
//  Potential Structs Definition
// └─────────────────────────────────────────────────────────┘
pub struct SHO;
pub struct DoubleWell;

#[allow(non_snake_case)]
pub struct Morse {
    a: f64,
    D_e: f64,
    r_e: f64,
}

#[allow(non_snake_case)]
pub struct Pendulum {
    theta_L: f64,
}

pub struct MirroredFreeFall;
pub struct SoftenedMirroredFreeFall;

// ┌─────────────────────────────────────────────────────────┐
//  Potential Implementations
// └─────────────────────────────────────────────────────────┘
// Simple Harmonic Oscillator
impl Potential for SHO {
    fn V(&self, q: f64) -> f64 {
        8f64 * (q - 0.5).powi(2)
    }

    fn dV(&self, q: f64) -> f64 {
        16f64 * (q - 0.5)
    }

    fn analytic_solution(&self, t: f64) -> Option<(f64, f64)> {
        let q = 0.5 - 0.5 * (4f64 * t).cos();
        let p = 2f64 * (4f64 * t).sin();
        Some((q, p))
    }
}

// Double Well Potential
impl Potential for DoubleWell {
    fn V(&self, q: f64) -> f64 {
        625f64 / 8f64 * (q - 0.2).powi(2) * (q - 0.8).powi(2)
    }

    fn dV(&self, q: f64) -> f64 {
        625f64 / 2f64 * q.powi(3) - 1875f64 / 4f64 * q.powi(2) + 825f64 / 4f64 * q - 25f64
    }

    fn analytic_solution(&self, _t: f64) -> Option<(f64, f64)> {
        None
    }
}

// Morse Potential
impl Potential for Morse {
    #[allow(non_snake_case)]
    fn V(&self, q: f64) -> f64 {
        self.D_e * (1f64 - (-self.a * (q - self.r_e)).exp()).powi(2)
    }

    #[allow(non_snake_case)]
    fn dV(&self, q: f64) -> f64 {
        2f64 * self.D_e
            * (1f64 - (-self.a * (q - self.r_e)).exp())
            * self.a
            * (-self.a * (q - self.r_e)).exp()
    }

    fn analytic_solution(&self, _t: f64) -> Option<(f64, f64)> {
        None
    }
}

// Pendulum Potential
impl Potential for Pendulum {
    fn V(&self, q: f64) -> f64 {
        let g = 2f64 / (1f64 - self.theta_L.cos());
        g * (1f64 - (2f64 * self.theta_L * (q - 0.5)).cos())
    }

    fn dV(&self, q: f64) -> f64 {
        let g = 2f64 / (1f64 - self.theta_L.cos());
        g * 2f64 * self.theta_L * (self.theta_L * (2f64 * q - 1f64)).sin()
    }

    fn analytic_solution(&self, _t: f64) -> Option<(f64, f64)> {
        None
    }
}

// Mirrored Free Fall Potential
impl Potential for MirroredFreeFall {
    fn V(&self, q: f64) -> f64 {
        4f64 * (q - 0.5).abs()
    }

    fn dV(&self, q: f64) -> f64 {
        4f64 * (q - 0.5).signum()
    }

    fn analytic_solution(&self, t: f64) -> Option<(f64, f64)> {
        if t < 0.5 {
            Some((2f64 * t.powi(2), 4f64 * t))
        } else if t < 1.5 {
            Some((-2f64 * t.powi(2) + 4f64 * t - 1f64, -4f64 * t + 4f64))
        } else {
            Some((2f64 * (2f64 - t).powi(2), -4f64 * (2f64 - t)))
        }
    }
}

// Softened Mirrored Free Fall Potential
impl Potential for SoftenedMirroredFreeFall {
    fn V(&self, q: f64) -> f64 {
        let alpha = 20f64;
        let a = 4f64 * (0.5 * alpha).tanh();
        a * (q - 0.5) / (alpha * (q - 0.5)).tanh()
    }

    fn dV(&self, q: f64) -> f64 {
        let alpha = 20f64;
        let a = 4f64 * (0.5 * alpha).tanh();
        a * (1f64 / (alpha * (q - 0.5)).tanh()
            - alpha * (q - 0.5) / (alpha * (q - 0.5)).sinh().powi(2))
    }

    fn analytic_solution(&self, _t: f64) -> Option<(f64, f64)> {
        None
    }
}

// ┌─────────────────────────────────────────────────────────┐
//  Data Structure
// └─────────────────────────────────────────────────────────┘
#[allow(non_snake_case)]
#[derive(Debug, Clone)]
pub struct Data {
    pub V: Vec<f64>,
    pub t: Vec<f64>,
    pub q: Vec<f64>,
    pub p: Vec<f64>,
}

// ┌─────────────────────────────────────────────────────────┐
//  Yoshida Solver
// └─────────────────────────────────────────────────────────┘
pub struct YoshidaSolver<V: Potential> {
    problem: V,
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

impl<V: Potential> YoshidaSolver<V> {
    pub fn new(problem: V) -> Self {
        YoshidaSolver { problem }
    }

    #[allow(non_snake_case)]
    pub fn solve(
        &self,
        t_span: (f64, f64),
        dt: f64,
        initial_condition: &[f64],
    ) -> anyhow::Result<Data> {
        let potential = &self.problem;

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
            // Yoshida 4th-order method
            for j in 0..4 {
                q = q + YOSHIDA_COEFF[j] * p * dt;
                p = p + YOSHIDA_COEFF[j + 4] * (-potential.dV(q)) * dt;
            }
            q_vec[i] = q;
            p_vec[i] = p;
        }

        let cs_q = cubic_hermite_spline(&t_vec, &q_vec, Quadratic)?;
        let cs_p = cubic_hermite_spline(&t_vec, &p_vec, Quadratic)?;

        let t_vec = linspace(t_span.0, t_span.1, NSENSORS);
        let q_vec = cs_q.eval_vec(&t_vec);
        let p_vec = cs_p.eval_vec(&t_vec);

        let q_uniform = linspace(-BOUNDARY, 1f64 + BOUNDARY, NSENSORS);
        let V = potential.V_map(&q_uniform);
        Ok(Data {
            V,
            t: t_vec,
            q: q_vec,
            p: p_vec,
        })
    }
}
