use peroxide::fuga::*;
use std::env;

const NSENSORS: usize = 100;

#[allow(non_snake_case)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let data_type = &args[1];
    let file_path = match data_type.as_str() {
        "normal" => "data_normal/train_cand.parquet",
        "more" => "data_more/train_cand.parquet",
        _ => {
            eprintln!("Usage: {} <data_type>", args[0]);
            eprintln!("data_type: 'normal' or 'more'");
            return Ok(());
        }
    };
    let df = DataFrame::read_parquet(file_path)?;

    let V: Vec<f64> = df["V"].to_vec();
    let p: Vec<f64> = df["p"].to_vec();
    let q: Vec<f64> = df["q"].to_vec();

    let data_len = V.len() / NSENSORS;

    let V = matrix(V, data_len, NSENSORS, Row);
    let p = matrix(p, data_len, NSENSORS, Row);
    let q = matrix(q, data_len, NSENSORS, Row);

    let q_domain = linspace(0.0, 1.0, NSENSORS);

    let mut E_delta_max = vec![0.0; data_len];

    for i in 0..data_len {
        let p_square = p.row(i).fmap(|p| p.powi(2) / 2f64);
        let q_raw = q.row(i);
        let potential_raw = V.row(i);
        let potential = cubic_hermite_spline(&q_domain, &potential_raw, Quadratic)?;
        let energy = potential.eval_vec(&q_raw).add_v(&p_square);
        let E_min = energy.min();
        let E_max = energy.max();
        E_delta_max[i] = (E_max - E_min) / (E_max + E_min).max(1e-10);
    }

    let mut df = DataFrame::new(vec![]);
    df.push("E_delta_max", Series::new(E_delta_max));
    df.print();
    df.write_parquet(&format!("data_analyze/{}_conserved.parquet", data_type), CompressionOptions::Snappy)?;

    Ok(())
}
