use dialoguer::{theme::ColorfulTheme, Input};

#[allow(non_snake_case)]
fn main() {
    let eta_0 = Input::<f64>::with_theme(&ColorfulTheme::default())
        .with_prompt("Input eta_0")
        .interact()
        .unwrap();

    let eta_inf = Input::<f64>::with_theme(&ColorfulTheme::default())
        .with_prompt("Input eta_inf")
        .interact()
        .unwrap();

    let eta_N = Input::<f64>::with_theme(&ColorfulTheme::default())
        .with_prompt("Input eta_N")
        .interact()
        .unwrap();

    let N = Input::<u64>::with_theme(&ColorfulTheme::default())
        .with_prompt("Input N")
        .interact()
        .unwrap();

    let U = Input::<u64>::with_theme(&ColorfulTheme::default())
        .with_prompt("Input U")
        .interact()
        .unwrap();

    let m = Input::<f64>::with_theme(&ColorfulTheme::default())
        .with_prompt("Input m")
        .interact()
        .unwrap();

    let n = Input::<f64>::with_theme(&ColorfulTheme::default())
        .with_prompt("Input n")
        .interact()
        .unwrap();

    let log_eta_0 = eta_0.log10();
    let log_eta_inf = eta_inf.log10();
    let log_eta_N = eta_N.log10();

    let new_log_eta_inf = new_eta_inf(log_eta_0, log_eta_inf, log_eta_N, N, U, m, n);
    let new_eta_inf = 10f64.powf(new_log_eta_inf);
    println!("new_eta_inf: {:.16e}", new_eta_inf);
}

#[allow(non_snake_case)]
fn new_eta_inf(eta_0: f64, eta_inf: f64, eta_N: f64, N: u64, U: u64, m: f64, n: f64) -> f64 {
    let N_f64 = N as f64;
    let U_f64 = U as f64;
    (m * eta_0 + n * eta_inf) / (m + n) + m / (m + n) * U_f64 / N_f64 * (eta_N - eta_0)
}
