use peroxide::fuga::*;
use dialoguer::{Select, theme::ColorfulTheme};
use pacmap::{Configuration, fit_transform};
use ndarray::Array2;

const N: usize = 100;

#[allow(non_snake_case)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_options = vec![
        "normal",
        "more",
        "much",
        "test",
    ];
    let selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select a dataset")
        .default(0)
        .items(&data_options)
        .interact()
        .unwrap();
    let data_type = data_options[selection];

    match data_type {
        "normal" | "more" | "much" => {
            let cand_file = format!("data_{}/train_cand.parquet", data_type);
            let df = DataFrame::read_parquet(&cand_file)?;
            df.print();

            let V_array = reshape_df(&df)?;
            println!("{}", V_array);
        },
        "test" => {
            let df = DataFrame::read_parquet("data_test/test_cand.parquet")?;
            df.print();

            let V_array = reshape_df(&df)?;
            println!("{}", V_array);

            let pacmap_df = pacmap_embedding(&df)?;
            let reshaped_df = reshape_array(&pacmap_df);
            reshaped_df.print();
        },
        _ => {
            eprintln!("Invalid selection");
            return Ok(());
        }
    }

    Ok(())
}

#[allow(non_snake_case)]
fn reshape_df(df: &DataFrame) -> anyhow::Result<Array2<f32>> {
    let V: Vec<f64> = df["V"].to_vec();
    let V = V.iter().map(|&x| x as f32).collect::<Vec<f32>>();
    let row = V.len() / N;

    let V_array = Array2::from_shape_vec((row, N), V)?;
    Ok(V_array)
}

#[allow(non_snake_case)]
fn reshape_array(array: &Array2<f32>) -> DataFrame {
    let pacmap_1 = array.column(0).iter().map(|&x| x as f64).collect::<Vec<f64>>();
    let pacmap_2 = array.column(1).iter().map(|&x| x as f64).collect::<Vec<f64>>();
    let mut df = DataFrame::new(vec![]);
    df.push("pacmap1", Series::new(pacmap_1));
    df.push("pacmap2", Series::new(pacmap_2));
    df
}

#[allow(non_snake_case)]
fn pacmap_embedding(df: &DataFrame) -> anyhow::Result<Array2<f32>> {
    let V_array = reshape_df(df)?;
    let config = Configuration::builder()
        .seed(42)
        .embedding_dimensions(2)
        .initialization(pacmap::Initialization::Random(Some(42)))
        .build();
    let (embedding, _) = fit_transform(V_array.view(), config)?;

    Ok(embedding)
}
