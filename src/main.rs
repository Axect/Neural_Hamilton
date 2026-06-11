use indicatif::{ParallelProgressIterator, ProgressBar, ProgressIterator};
use peroxide::fuga::*;
use rayon::prelude::*;
use rugfield::{grf_with_rng, Kernel};
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

const V0: f64 = 2f64;
const CHUNK_SIZE: usize = 10_000; // Process 10k potentials at a time to limit memory
const L: f64 = 1f64;
const NSENSORS: usize = 100;
const BOUNDARY: f64 = 0f64;
const TSTEP: f64 = 1e-3;
const T_WINDOW: f64 = 2f64; // Time window size for each sample

// --- Phase-space IC sampling (v1.33) ---
// Each potential yields NDIFFCONFIG samples, one per energy stratum.
// The stratum variable is u = (E - V(q0)) / (V0 - V(q0)) in (0, 1):
// low-u orbits librate near the local minimum containing q0 (sub-well motion
// for multi-well potentials), high-u orbits approach the full-traversal
// regime of the legacy dataset (E ~= V0, turning points near the walls).
const NDIFFCONFIG: usize = 4; // Number of energy strata (= samples) per potential
const ENERGY_STRATA: [(f64, f64); NDIFFCONFIG] = [
    (0.05, 0.35),
    (0.35, 0.65),
    (0.65, 0.95),
    (0.95, 0.99),
];

// Per-sample depth scaling: interior control values are mapped to
// [V0 - depth, V0 - HEADROOM] with depth ~ U(DEPTH_MIN, DEPTH_MAX).
// HEADROOM keeps the walls strictly highest, so any orbit with E < V0 is
// confined to [0, L] by construction (no boundary rejection bias).
const DEPTH_MIN: f64 = 0.5;
const DEPTH_MAX: f64 = 3.5;
const HEADROOM: f64 = 0.1;

// Energy drift tolerance, relative to the per-potential energy scale (~depth)
const E_DRIFT_TOL: f64 = 1e-4;

// Target data counts (samples = potentials x NDIFFCONFIG)
const TARGET_TRAIN: usize = 16_000;
const TARGET_VAL: usize = 4_000;
const TARGET_TEST: usize = 20_000;
const SAFETY_FACTOR: f64 = 2.5; // Generate 2.5x candidates to account for trajectory filtering
const DIVERSITY_FACTOR: f64 = 2.5; // GRF oversample for diversity selection before trajectories
const N_BINS: usize = 8; // Bins per feature dimension for diversity hashing

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let selection = if args.len() > 1 {
        args[1].parse::<usize>().unwrap_or(0)
    } else {
        println!("No selection provided, defaulting to Normal mode.");
        0
    };

    match selection {
        0 => {
            // Normal mode (Train/Val) - loop until target is reached
            let (target_train, target_val, folder) = (TARGET_TRAIN, TARGET_VAL, "data_normal");

            println!("\n=== Generate training data (target: {}) ===", target_train);
            let ds_train = Dataset::generate_loop(target_train, 123)?;
            println!("Training data: {} (exact)", ds_train.data.len());
            if !ds_train.data.is_empty() {
                let (q_max, p_max) = ds_train.max();
                println!("Max of q: {:.4}, p: {:.4}", q_max, p_max);
            }
            ds_train.write_parquet(&format!("{}/train.parquet", folder))?;

            println!("\n=== Generate validation data (target: {}) ===", target_val);
            let ds_val = Dataset::generate_loop(target_val, 456)?;
            println!("Validation data: {} (exact)", ds_val.data.len());
            if !ds_val.data.is_empty() {
                let (q_max, p_max) = ds_val.max();
                println!("Max of q: {:.4}, p: {:.4}", q_max, p_max);
            }
            ds_val.write_parquet(&format!("{}/val.parquet", folder))?;
        }
        1 => {
            // More mode (10x data) - chunked loop to avoid OOM
            let target_train = TARGET_TRAIN * 10;
            let target_val = TARGET_VAL * 10;
            let folder = "data_more";

            println!("\n=== Chunked generation for large dataset ===");
            println!("Chunk size: {} potentials per chunk", CHUNK_SIZE);

            println!("\n=== Generate training data (target: {}) ===", target_train);
            let train_count = Dataset::generate_chunked_loop(
                target_train,
                123,
                &format!("{}/train.parquet", folder),
            )?;
            println!("Training data: {} (exact)", train_count);

            println!("\n=== Generate validation data (target: {}) ===", target_val);
            let val_count = Dataset::generate_chunked_loop(
                target_val,
                456,
                &format!("{}/val.parquet", folder),
            )?;
            println!("Validation data: {} (exact)", val_count);
        }
        2 => {
            // Test - loop until target is reached
            let target_test = TARGET_TEST;
            let folder = "data_test";

            println!("\n=== Generate test data (target: {}) ===", target_test);
            let ds_test = Dataset::generate_loop(target_test, 8407)?;
            println!("Test data: {} (exact)", ds_test.data.len());
            if !ds_test.data.is_empty() {
                let (q_max, p_max) = ds_test.max();
                println!("Max of q: {:.4}, p: {:.4}", q_max, p_max);
            }
            ds_test.write_parquet(&format!("{}/test.parquet", folder))?;
        }
        3 => {
            // Smoke test - small dataset for pipeline validation
            let target = if args.len() > 2 {
                args[2].parse::<usize>().unwrap_or(400)
            } else {
                400
            };
            let folder = "data_smoke";

            println!("\n=== Generate smoke-test data (target: {}) ===", target);
            let ds = Dataset::generate_loop(target, 777)?;
            println!("Smoke data: {} (exact)", ds.data.len());
            if !ds.data.is_empty() {
                let (q_max, p_max) = ds.max();
                println!("Max of q: {:.4}, p: {:.4}", q_max, p_max);
            }
            ds.write_parquet(&format!("{}/smoke.parquet", folder))?;
        }
        _ => unreachable!(),
    }
    Ok(())
}

/// Parquet column layout (flat format, NSENSORS rows per sample):
/// V, t, q, p are per-row values; the remaining columns are per-sample
/// metadata repeated NSENSORS times.
const COLUMNS: [&str; 10] = ["V", "t", "q", "p", "pid", "b", "l", "depth", "E0", "u"];

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

    /// Generate data in a loop until the target count is reached.
    /// Automatically retries with new seeds if the trajectory filters
    /// reject too many candidates.
    pub fn generate_loop(target: usize, initial_seed: u64) -> anyhow::Result<Self> {
        let mut all_data: Vec<Data> = Vec::new();
        let mut seed = initial_seed;
        let mut attempt = 0;
        // pid space advances by the number of candidates handed to the
        // trajectory stage (including rejected ones), so pids never collide
        // across attempts even when filters drop candidates.
        let mut pid_offset: usize = 0;

        while all_data.len() < target {
            attempt += 1;
            let remaining = target - all_data.len();
            let n_cand = (remaining as f64 * SAFETY_FACTOR).ceil() as usize;

            println!(
                "\n[Attempt {}, seed: {}] Need {} more, generating {} candidates...",
                attempt, seed, remaining, n_cand
            );

            let potential_generator = BoundedPotential::generate_potential(n_cand, seed);
            let n_candidates = potential_generator.candidates.len();
            let ds = BoundedPotential::generate_data_from_potentials(
                &potential_generator.candidates,
                pid_offset,
            )?;
            pid_offset += n_candidates;
            let generated = ds.data.len();
            all_data.extend(ds.data);

            println!(
                "[Attempt {}] Got {} → total {}/{}",
                attempt,
                generated,
                all_data.len(),
                target
            );

            seed += 100_000;
        }

        all_data.truncate(target);
        Ok(Dataset::new(all_data))
    }

    /// Generate data in chunks with automatic retry until target is reached.
    /// Writes chunks to disk to avoid OOM for large datasets.
    /// Returns the total number of generated samples.
    pub fn generate_chunked_loop(
        target: usize,
        initial_seed: u64,
        output_path: &str,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let parent = Path::new(output_path).parent().unwrap();
        if !parent.exists() {
            std::fs::create_dir_all(parent)?;
        }

        let mut total_generated: usize = 0;
        let mut chunk_files: Vec<String> = Vec::new();
        let mut seed = initial_seed;
        let mut attempt = 0;
        // See generate_loop: advance by candidates processed, not samples kept
        let mut pid_offset: usize = 0;

        while total_generated < target {
            attempt += 1;
            let remaining = target - total_generated;
            let n_cand = (remaining as f64 * SAFETY_FACTOR).ceil() as usize;

            println!(
                "\n[Attempt {}, seed: {}] Need {} more, generating {} candidates...",
                attempt, seed, remaining, n_cand
            );

            let potential_generator = BoundedPotential::generate_potential(n_cand, seed);
            let total_potentials = potential_generator.candidates.len();
            let num_chunks = (total_potentials + CHUNK_SIZE - 1) / CHUNK_SIZE;

            for chunk_idx in 0..num_chunks {
                let start = chunk_idx * CHUNK_SIZE;
                let end = std::cmp::min(start + CHUNK_SIZE, total_potentials);
                let chunk_candidates: Vec<_> =
                    potential_generator.candidates[start..end].to_vec();

                let chunk_data = BoundedPotential::generate_data_from_potentials(
                    &chunk_candidates,
                    pid_offset,
                )
                .map_err(|e| -> Box<dyn std::error::Error> {
                    Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        e.to_string(),
                    ))
                })?;
                pid_offset += chunk_candidates.len();
                let chunk_count = chunk_data.data.len();
                total_generated += chunk_count;

                // Skip writing empty chunks (can happen with strict filters)
                if chunk_count > 0 {
                    let chunk_path =
                        format!("{}.chunk_{:04}.parquet", output_path, chunk_files.len());
                    chunk_data.write_parquet(&chunk_path)?;
                    chunk_files.push(chunk_path);
                }

                println!(
                    "Chunk: {} samples (total: {}/{})",
                    chunk_count, total_generated, target
                );

                if total_generated >= target {
                    break;
                }
            }

            seed += 100_000;
        }

        // Concatenate all chunk files into final output
        println!("\nConcatenating {} chunk files...", chunk_files.len());
        Self::concatenate_parquet_files(&chunk_files, output_path, target)?;

        // Clean up chunk files
        println!("Cleaning up temporary chunk files...");
        for chunk_file in &chunk_files {
            if Path::new(chunk_file).exists() {
                std::fs::remove_file(chunk_file)?;
            }
        }

        println!("Final output written to: {}", output_path);
        Ok(std::cmp::min(total_generated, target))
    }

    /// Concatenate multiple parquet files into one, taking only up to `target` samples
    fn concatenate_parquet_files(
        chunk_files: &[String],
        output_path: &str,
        target: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut all_cols: Vec<Vec<f64>> = vec![Vec::new(); COLUMNS.len()];

        let mut rows_collected = 0;
        let rows_per_sample = NSENSORS;

        for chunk_file in chunk_files {
            if rows_collected >= target {
                break;
            }

            let df = DataFrame::read_parquet(chunk_file)?;
            let cols: Vec<Vec<f64>> = COLUMNS.iter().map(|&c| df[c].to_vec()).collect();

            let samples_in_chunk = cols[0].len() / rows_per_sample;
            let samples_needed = target - rows_collected;
            let samples_to_take = std::cmp::min(samples_in_chunk, samples_needed);
            let rows_to_take = samples_to_take * rows_per_sample;

            for (acc, col) in all_cols.iter_mut().zip(cols.iter()) {
                acc.extend_from_slice(&col[..rows_to_take]);
            }

            rows_collected += samples_to_take;
        }

        let mut df = DataFrame::new(vec![]);
        for (&name, col) in COLUMNS.iter().zip(all_cols.into_iter()) {
            df.push(name, Series::new(col));
        }

        println!(
            "Final dataset: {} samples ({} rows)",
            rows_collected,
            rows_collected * rows_per_sample
        );
        df.write_parquet(output_path, SNAPPY)?;

        Ok(())
    }

    #[allow(non_snake_case)]
    pub fn write_parquet(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let parent = std::path::Path::new(path).parent().unwrap();
        if !parent.exists() {
            std::fs::create_dir_all(parent)?;
        }

        let n_rows = self.data.len() * NSENSORS;
        let mut cols: Vec<Vec<f64>> = (0..COLUMNS.len())
            .map(|_| Vec::with_capacity(n_rows))
            .collect();

        for d in &self.data {
            cols[0].extend_from_slice(&d.V);
            cols[1].extend_from_slice(&d.t);
            cols[2].extend_from_slice(&d.q);
            cols[3].extend_from_slice(&d.p);
            for (i, &v) in [d.pid, d.b, d.l, d.depth, d.e0, d.u].iter().enumerate() {
                cols[4 + i].extend(std::iter::repeat(v).take(NSENSORS));
            }
        }

        let mut df = DataFrame::new(vec![]);
        for (&name, col) in COLUMNS.iter().zip(cols.into_iter()) {
            df.push(name, Series::new(col));
        }
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
    // Per-sample metadata (repeated per row in the flat parquet format)
    pub pid: f64,   // unique potential id within the output file
    pub b: f64,     // number of GRF control points
    pub l: f64,     // GRF length scale
    pub depth: f64, // potential depth D: interior values lie in [V0 - D, V0 - HEADROOM]
    pub e0: f64,    // orbit energy H(q0, p0)
    pub u: f64,     // energy fraction (E0 - V(q0)) / (V0 - V(q0))
}

/// A generated potential candidate with its GRF parameters.
#[derive(Debug, Clone)]
pub struct PotentialCandidate {
    pub q: Vec<f64>,
    pub v: Vec<f64>,
    pub b: usize,
    pub l: f64,
    pub depth: f64,
}

#[derive(Debug, Clone)]
pub struct BoundedPotential {
    pub candidates: Vec<PotentialCandidate>,
}

impl BoundedPotential {
    #[allow(non_snake_case)]
    pub fn generate_potential(n: usize, seed: u64) -> Self {
        let n = n / NDIFFCONFIG; // Unique potentials needed for n samples

        // Generate more GRFs than needed so the diversity filter has candidates to choose from
        let n_grf = (n as f64 * DIVERSITY_FACTOR * 1.1).round() as usize;
        let omega = 0.05;
        let degree = 3;
        let mut rng = stdrng_from_seed(seed);

        // Use LHS for better parameter coverage instead of uniform sampling
        let bl_pairs = lhs_sample_bl(n_grf, &mut rng);
        let b: Vec<usize> = bl_pairs.iter().map(|&(b, _)| b).collect();
        let l: Vec<f64> = bl_pairs.iter().map(|&(_, l)| l).collect();

        println!(
            "Generating {} GRF potentials (LHS) to select {} diverse...",
            n_grf, n
        );

        let grf_vec = b
            .iter()
            .zip(l.iter())
            .progress_with(ProgressBar::new(n_grf as u64))
            .map(|(&b, &l)| grf_with_rng(&mut rng, b, Kernel::SquaredExponential(l)))
            .collect::<Vec<_>>();

        // Per-sample depth: interior control values mapped to [V0 - depth, V0 - HEADROOM].
        // Explicit depth sampling replaces the former batch-global min-max scaling,
        // which made the depth distribution depend on batch composition.
        let depth_uniform = Uniform(DEPTH_MIN, DEPTH_MAX);
        let depth_vec: Vec<f64> = depth_uniform.sample_with_rng(&mut rng, n_grf);

        let mut grf_scaled_vec = grf_vec
            .par_iter()
            .zip(depth_vec.par_iter())
            .map(|(grf, &depth)| {
                let gmin = grf.min();
                let gmax = grf.max();
                let range = (gmax - gmin).max(1e-12);
                grf.fmap(|x| (V0 - depth) + (depth - HEADROOM) * (x - gmin) / range)
            })
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

        let all_candidates: Vec<PotentialCandidate> = (0..q_vec.len())
            .into_par_iter()
            .progress_with(ProgressBar::new(n_grf as u64))
            .filter_map(|i| {
                let q_coords = &q_vec[i];
                let V_coords = &grf_scaled_vec[i];
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
                        Some(PotentialCandidate {
                            q: q_new,
                            v: potential_new,
                            b: b[i],
                            l: l[i],
                            depth: depth_vec[i],
                        })
                    }
                    Err(_) => None,
                }
            })
            .collect();

        println!(
            "B-spline creation: {} succeeded out of {} GRFs",
            all_candidates.len(),
            n_grf
        );

        // Apply diversity filter to select n potentials from the candidates
        let selected_indices = diversity_filter(&all_candidates, n, seed + 1000);
        let candidates: Vec<PotentialCandidate> = selected_indices
            .into_iter()
            .map(|i| all_candidates[i].clone())
            .collect();

        println!("After diversity filter: {} potentials selected", candidates.len());

        BoundedPotential { candidates }
    }

    /// Generate data from a slice of potential candidates.
    /// For each potential, sample NDIFFCONFIG initial conditions (one per
    /// energy stratum) and integrate each orbit over [0, T_WINDOW].
    #[allow(non_snake_case)]
    pub fn generate_data_from_potentials(
        candidates: &[PotentialCandidate],
        pid_offset: usize,
    ) -> anyhow::Result<Dataset> {
        let q_domain = linspace(0f64, L, NSENSORS);

        // Rejection counters for pipeline transparency
        let n_spline_fail = AtomicUsize::new(0);
        let n_vgrid_bad = AtomicUsize::new(0);
        let n_ic_fail = AtomicUsize::new(0);
        let n_integ_fail = AtomicUsize::new(0);
        let n_bound_fail = AtomicUsize::new(0);
        let n_edrift_fail = AtomicUsize::new(0);

        let data_vec: Vec<Data> = candidates
            .par_iter()
            .enumerate()
            .progress_with(ProgressBar::new(candidates.len() as u64))
            .filter_map(|(idx, cand)| {
                let hamilton_eq =
                    match HamiltonEquation::new((cand.q.clone(), cand.v.clone())) {
                        Ok(eq) => eq,
                        Err(_) => {
                            n_spline_fail.fetch_add(1, Ordering::Relaxed);
                            return None;
                        }
                    };

                // Potential values at the uniform sensor grid (model input)
                let V_grid = hamilton_eq.eval_vec(&q_domain);
                if V_grid.iter().any(|&x| !x.is_finite() || x.abs() > 10f64) {
                    n_vgrid_bad.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
                let v_min = V_grid.iter().cloned().fold(f64::INFINITY, f64::min);
                let energy_scale = (V0 - v_min).max(DEPTH_MIN);

                // Deterministic RNG seeded from potential data for reproducibility
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};

                let mut hasher = DefaultHasher::new();
                for &v in &V_grid {
                    v.to_bits().hash(&mut hasher);
                }
                let seed = hasher.finish();
                let mut rng = stdrng_from_seed(seed);

                let mut windows: Vec<Data> = Vec::with_capacity(NDIFFCONFIG);
                for k in 0..NDIFFCONFIG {
                    // Sample IC on energy stratum k
                    let (q0, p0, e0, u) = match sample_ic(&hamilton_eq, k, &mut rng) {
                        Some(ic) => ic,
                        None => {
                            n_ic_fail.fetch_add(1, Ordering::Relaxed);
                            return None;
                        }
                    };

                    // Integrate orbit with Yoshida 8th order
                    let (t_dense, q_dense, p_dense) =
                        match integrate_window(&hamilton_eq, q0, p0) {
                            Some(sol) => sol,
                            None => {
                                n_integ_fail.fetch_add(1, Ordering::Relaxed);
                                return None;
                            }
                        };

                    // Confinement check: E0 < V0 guarantees q in [0, L] analytically;
                    // this catches numerical violations only.
                    if q_dense
                        .iter()
                        .any(|&x| x < -BOUNDARY || x > L + BOUNDARY)
                    {
                        n_bound_fail.fetch_add(1, Ordering::Relaxed);
                        return None;
                    }

                    // Energy conservation check on dense solver points
                    let e_iter = q_dense
                        .iter()
                        .zip(p_dense.iter())
                        .map(|(&q, &p)| hamilton_eq.eval(q) + p * p / 2f64);
                    let (e_min, e_max) = e_iter.fold(
                        (f64::INFINITY, f64::NEG_INFINITY),
                        |(lo, hi), e| (lo.min(e), hi.max(e)),
                    );
                    if (e_max - e_min) >= E_DRIFT_TOL * energy_scale {
                        n_edrift_fail.fetch_add(1, Ordering::Relaxed);
                        return None;
                    }

                    // Interpolate and sample at random time points
                    let cs_q = match cubic_hermite_spline(&t_dense, &q_dense, Quadratic) {
                        Ok(s) => s,
                        Err(_) => {
                            n_spline_fail.fetch_add(1, Ordering::Relaxed);
                            return None;
                        }
                    };
                    let cs_p = match cubic_hermite_spline(&t_dense, &p_dense, Quadratic) {
                        Ok(s) => s,
                        Err(_) => {
                            n_spline_fail.fetch_add(1, Ordering::Relaxed);
                            return None;
                        }
                    };

                    let mut t_domain =
                        Uniform(0f64, T_WINDOW).sample_with_rng(&mut rng, NSENSORS - 2);
                    t_domain.insert(0, 0f64);
                    t_domain.push(T_WINDOW);
                    t_domain.sort_by(|a, b| a.partial_cmp(b).unwrap());

                    let q_out = cs_q.eval_vec(&t_domain);
                    let p_out = cs_p.eval_vec(&t_domain);
                    if q_out.iter().any(|&x| !x.is_finite())
                        || p_out.iter().any(|&x| !x.is_finite())
                    {
                        n_integ_fail.fetch_add(1, Ordering::Relaxed);
                        return None;
                    }

                    windows.push(Data {
                        V: V_grid.clone(),
                        t: t_domain,
                        q: q_out,
                        p: p_out,
                        pid: (pid_offset + idx) as f64,
                        b: cand.b as f64,
                        l: cand.l,
                        depth: cand.depth,
                        e0,
                        u,
                    });
                }

                Some(windows)
            })
            .flatten()
            .collect();

        println!(
            "Generated data: {} samples (from {} potentials); rejects: spline {}, V-grid {}, IC {}, integration {}, bound {}, E-drift {}",
            data_vec.len(),
            data_vec.len() / NDIFFCONFIG,
            n_spline_fail.load(Ordering::Relaxed),
            n_vgrid_bad.load(Ordering::Relaxed),
            n_ic_fail.load(Ordering::Relaxed),
            n_integ_fail.load(Ordering::Relaxed),
            n_bound_fail.load(Ordering::Relaxed),
            n_edrift_fail.load(Ordering::Relaxed),
        );
        Ok(Dataset::new(data_vec))
    }
}

/// Sample an initial condition (q0, p0) on energy stratum `k`.
/// Returns (q0, p0, E0, u) with E0 = V(q0) + u * (V0 - V(q0)).
fn sample_ic(
    hamilton_eq: &HamiltonEquation,
    k: usize,
    rng: &mut StdRng,
) -> Option<(f64, f64, f64, f64)> {
    let (u_lo, u_hi) = ENERGY_STRATA[k];
    for _ in 0..100 {
        let q0 = Uniform(0f64, L).sample_with_rng(rng, 1)[0];
        let vq0 = hamilton_eq.eval(q0);
        let gap = V0 - vq0;
        // Reject points too close to wall energy (degenerate, near-zero p orbits
        // pinned at the wall) — interior headroom makes this rare.
        if gap < 1e-3 {
            continue;
        }
        let u = Uniform(u_lo, u_hi).sample_with_rng(rng, 1)[0];
        let e0 = vq0 + u * gap;
        let p_mag = (2f64 * (e0 - vq0)).sqrt();
        let sign = if Uniform(0f64, 1f64).sample_with_rng(rng, 1)[0] < 0.5 {
            -1f64
        } else {
            1f64
        };
        return Some((q0, sign * p_mag, e0, u));
    }
    None
}

/// Integrate Hamilton's equations from (q0, p0) over [0, T_WINDOW] with
/// the Yoshida 8th-order symplectic integrator at step TSTEP.
fn integrate_window(
    hamilton_eq: &HamiltonEquation,
    q0: f64,
    p0: f64,
) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let num_points = (T_WINDOW / TSTEP).round() as usize + 1;
    let t_vec = linspace(0f64, T_WINDOW, num_points);
    let mut q_vec = vec![0f64; num_points];
    let mut p_vec = vec![0f64; num_points];
    q_vec[0] = q0;
    p_vec[0] = p0;

    for i in 1..num_points {
        let mut q = q_vec[i - 1];
        let mut p = p_vec[i - 1];
        yoshida_step(hamilton_eq, 8, &mut q, &mut p, TSTEP);
        if !q.is_finite() || !p.is_finite() || q < -0.5 || q > L + 0.5 {
            return None;
        }
        q_vec[i] = q;
        p_vec[i] = p;
    }

    Some((t_vec, q_vec, p_vec))
}

/// Recursive Suzuki-Yoshida triple-jump composition.
///
/// S_2(h): Leapfrog (Störmer-Verlet, 2nd order symplectic)
/// S_{2n}(h) = S_{2n-2}(z1*h) ∘ S_{2n-2}(z0*h) ∘ S_{2n-2}(z1*h)
/// where z1 = 1/(2 - 2^{1/(2n-1)}), z0 = 1 - 2*z1
///
/// For 8th order: 27 force evaluations per step (3^3 leapfrogs).
fn yoshida_step(hamilton_eq: &HamiltonEquation, order: usize, q: &mut f64, p: &mut f64, dt: f64) {
    if order == 2 {
        // Leapfrog (Position Verlet): drift q, kick p, drift q
        *q += 0.5 * *p * dt;
        *p -= hamilton_eq.eval_grad(*q) * dt;
        *q += 0.5 * *p * dt;
    } else {
        let w = 2f64.powf(1.0 / (order as f64 - 1.0));
        let z1 = 1.0 / (2.0 - w);
        let z0 = -w / (2.0 - w);
        yoshida_step(hamilton_eq, order - 2, q, p, z1 * dt);
        yoshida_step(hamilton_eq, order - 2, q, p, z0 * dt);
        yoshida_step(hamilton_eq, order - 2, q, p, z1 * dt);
    }
}

// ============================================================
// Diversity filtering: LHS sampling + feature-hashing filter
// ============================================================

/// Lightweight statistical features extracted from a potential V(x).
/// Used for diversity filtering before expensive trajectory simulation.
struct PotentialFeatures {
    mean: f64,
    variance: f64,
    skewness: f64,
    kurtosis: f64,
    n_local_minima: f64,
    n_local_maxima: f64,
    mean_abs_gradient: f64,
}

impl PotentialFeatures {
    /// Extract 7 statistical features from a potential V(x) evaluated at uniform grid.
    fn extract(v: &[f64]) -> Self {
        let n = v.len() as f64;

        // Mean
        let mean = v.iter().sum::<f64>() / n;

        // Variance
        let variance = v.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt().max(1e-15);

        // Skewness & kurtosis (standardized moments)
        let skewness = v.iter().map(|&x| ((x - mean) / std_dev).powi(3)).sum::<f64>() / n;
        let kurtosis = v.iter().map(|&x| ((x - mean) / std_dev).powi(4)).sum::<f64>() / n;

        // Count local minima and maxima
        let mut n_local_minima = 0usize;
        let mut n_local_maxima = 0usize;
        for i in 1..v.len() - 1 {
            if v[i] < v[i - 1] && v[i] < v[i + 1] {
                n_local_minima += 1;
            }
            if v[i] > v[i - 1] && v[i] > v[i + 1] {
                n_local_maxima += 1;
            }
        }

        // Mean absolute gradient (finite differences)
        let mean_abs_gradient = v
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .sum::<f64>()
            / (n - 1.0);

        PotentialFeatures {
            mean,
            variance,
            skewness,
            kurtosis,
            n_local_minima: n_local_minima as f64,
            n_local_maxima: n_local_maxima as f64,
            mean_abs_gradient,
        }
    }

    fn as_array(&self) -> [f64; 7] {
        [
            self.mean,
            self.variance,
            self.skewness,
            self.kurtosis,
            self.n_local_minima,
            self.n_local_maxima,
            self.mean_abs_gradient,
        ]
    }
}

/// Latin Hypercube Sampling for (b, l) parameter pairs.
///
/// - `b` is discrete in {2, 3, 4, 5, 6, 7}: equal allocation across 6 values
/// - `l` is continuous in [0.01, 0.2]: stratified within each b-group
///
/// Returns Vec<(usize, f64)> of (b, l) pairs.
fn lhs_sample_bl(n: usize, rng: &mut StdRng) -> Vec<(usize, f64)> {
    let b_values: Vec<usize> = vec![2, 3, 4, 5, 6, 7];
    let n_b = b_values.len();
    let per_b = n / n_b;
    let remainder = n % n_b;

    let l_min = 0.01f64;
    let l_max = 0.2f64;

    let mut pairs: Vec<(usize, f64)> = Vec::with_capacity(n);

    for (i, &b_val) in b_values.iter().enumerate() {
        let count = if i < remainder { per_b + 1 } else { per_b };
        if count == 0 {
            continue;
        }
        let stratum_width = (l_max - l_min) / count as f64;

        for j in 0..count {
            // Stratified sample within [l_min + j*w, l_min + (j+1)*w]
            let lo = l_min + j as f64 * stratum_width;
            let hi = lo + stratum_width;
            let u = Uniform(lo, hi);
            let l_val = u.sample_with_rng(rng, 1)[0];
            pairs.push((b_val, l_val));
        }
    }

    // Fisher-Yates shuffle for random ordering
    let len = pairs.len();
    for i in (1..len).rev() {
        let j = Uniform(0.0, (i + 1) as f64).sample_with_rng(rng, 1)[0] as usize;
        pairs.swap(i, j);
    }

    pairs
}

/// Diversity filter using feature-hashing with binned features.
///
/// Algorithm:
/// 1. Extract features for all potentials (parallel)
/// 2. Min-max normalize each feature
/// 3. Quantize to N_BINS bins per feature → pack into u64 key
/// 4. Group by bin key, allocate inversely proportional to density
/// 5. Deterministic sampling within each bin to hit exact target count
///
/// Returns indices of selected potentials.
fn diversity_filter(candidates: &[PotentialCandidate], target: usize, seed: u64) -> Vec<usize> {
    if candidates.len() <= target {
        return (0..candidates.len()).collect();
    }

    // 1. Extract features in parallel
    let features: Vec<[f64; 7]> = candidates
        .par_iter()
        .map(|c| PotentialFeatures::extract(&c.v).as_array())
        .collect();

    // 2. Compute min/max per feature for normalization
    let n_features = 7;
    let mut feat_min = [f64::INFINITY; 7];
    let mut feat_max = [f64::NEG_INFINITY; 7];
    for f in &features {
        for j in 0..n_features {
            if f[j] < feat_min[j] {
                feat_min[j] = f[j];
            }
            if f[j] > feat_max[j] {
                feat_max[j] = f[j];
            }
        }
    }

    // 3. Quantize to bins and pack into u64 key
    //    With 7 features × 8 bins = 7×3 = 21 bits, fits easily in u64
    let bin_keys: Vec<u64> = features
        .iter()
        .map(|f| {
            let mut key: u64 = 0;
            for j in 0..n_features {
                let range = feat_max[j] - feat_min[j];
                let normalized = if range > 1e-15 {
                    ((f[j] - feat_min[j]) / range).clamp(0.0, 1.0 - 1e-10)
                } else {
                    0.5
                };
                let bin = (normalized * N_BINS as f64) as u64;
                key = key * N_BINS as u64 + bin;
            }
            key
        })
        .collect();

    // 4. Group indices by bin key
    let mut bins: HashMap<u64, Vec<usize>> = HashMap::new();
    for (idx, &key) in bin_keys.iter().enumerate() {
        bins.entry(key).or_default().push(idx);
    }

    let n_occupied = bins.len();
    let total = candidates.len();

    println!(
        "Diversity filter: {} potentials → {} target, {} occupied bins",
        total, target, n_occupied
    );

    // 5. Allocate inversely proportional to density
    //    Weight of bin = 1 / count_in_bin → normalized so sum = target
    let total_weight: f64 = bins.values().map(|v| 1.0 / v.len() as f64).sum();
    let mut allocations: Vec<(u64, usize)> = bins
        .iter()
        .map(|(&key, members)| {
            let weight = 1.0 / members.len() as f64;
            let alloc = ((weight / total_weight) * target as f64).floor() as usize;
            // Don't allocate more than available
            (key, alloc.min(members.len()))
        })
        .collect();

    // Sort by key for determinism
    allocations.sort_by_key(|&(key, _)| key);

    // Greedy adjustment to hit exact target
    let mut current_total: usize = allocations.iter().map(|(_, a)| a).sum();

    // If we're short, add one to bins with lowest density (most room) first
    if current_total < target {
        // Sort bins by density ascending (sparse bins get priority)
        let mut bin_density: Vec<(u64, f64, usize)> = allocations
            .iter()
            .map(|&(key, alloc)| {
                let cap = bins[&key].len();
                let density = cap as f64;
                (key, density, alloc)
            })
            .collect();
        bin_density.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut alloc_map: HashMap<u64, usize> = allocations.iter().cloned().collect();

        for (key, _density, _) in &bin_density {
            if current_total >= target {
                break;
            }
            let cap = bins[key].len();
            let current = alloc_map[key];
            if current < cap {
                *alloc_map.get_mut(key).unwrap() += 1;
                current_total += 1;
            }
        }

        // If still short after one pass, keep cycling
        while current_total < target {
            let mut added = false;
            for (key, _density, _) in &bin_density {
                if current_total >= target {
                    break;
                }
                let cap = bins[key].len();
                let current = alloc_map[key];
                if current < cap {
                    *alloc_map.get_mut(key).unwrap() += 1;
                    current_total += 1;
                    added = true;
                }
            }
            if !added {
                break; // All bins full
            }
        }

        allocations = alloc_map.into_iter().collect();
        allocations.sort_by_key(|&(key, _)| key);
    }

    // If we're over, trim from densest bins
    while current_total > target {
        // Find bin with largest allocation that has allocation > 0
        if let Some(pos) = allocations
            .iter()
            .enumerate()
            .filter(|(_, (_, a))| *a > 0)
            .max_by_key(|(_, (key, _))| bins[key].len())
            .map(|(i, _)| i)
        {
            allocations[pos].1 -= 1;
            current_total -= 1;
        } else {
            break;
        }
    }

    // 6. Deterministic sampling within each bin
    let mut rng = stdrng_from_seed(seed);
    let mut selected: Vec<usize> = Vec::with_capacity(target);

    for (key, alloc) in &allocations {
        if *alloc == 0 {
            continue;
        }
        let members = &bins[key];
        if *alloc >= members.len() {
            selected.extend(members);
        } else {
            // Reservoir sampling with deterministic RNG
            let mut candidates_in_bin = members.clone();
            // Partial Fisher-Yates to select `alloc` elements
            for i in 0..*alloc {
                let j = i + (Uniform(0.0, (candidates_in_bin.len() - i) as f64)
                    .sample_with_rng(&mut rng, 1)[0] as usize);
                candidates_in_bin.swap(i, j);
            }
            selected.extend_from_slice(&candidates_in_bin[..*alloc]);
        }
    }

    // Sort selected indices for stable ordering
    selected.sort_unstable();
    selected
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
