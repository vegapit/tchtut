#![allow(non_snake_case)]
extern crate plotters;
extern crate tchtut;

use plotters::prelude::*;
use tchtut::{KuhnPoker,PolicyGradientAlgorithm};

fn plot(data1: Vec<(f32,f32)>,data2: Vec<(f32,f32)>) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("result.png", (1024, 768)).into_drawing_area();
    
    root.fill(&WHITE)?;
    
    let root = root
        .margin(0, 0, 0, 20);

    let areas = root.split_evenly((2,1));

    let mut chart1 = ChartBuilder::on(&areas[0])
        .x_label_area_size(50)
        .y_label_area_size(60)
        .caption("Average Critic Loss", ("Arial", 40).into_font())
        .build_ranged( 0f32..(data1.len() as f32 + 1f32), 0f32..5f32)?;

    chart1.configure_mesh().draw()?;

    chart1.draw_series(
        LineSeries::new(data1, &BLUE)
    )?
        .label("Critic")
        .legend( |(x,y)| Path::new(vec![(x,y), (x + 20,y)], &BLUE) );

    chart1.configure_series_labels()
        .background_style( &WHITE.mix(0.8) )
        .border_style( &BLACK )
        .draw()?;

    let mut chart2 = ChartBuilder::on(&areas[1])
        .x_label_area_size(50)
        .y_label_area_size(60)
        .caption("Average Actor Loss", ("Arial", 40).into_font())
        .build_ranged( 0f32..(data2.len() as f32 + 1f32), -0.2f32..0.2f32)?;

    chart2.configure_mesh().draw()?;

    chart2.draw_series(
        LineSeries::new(data2, &RED)
    )?
        .label("Actor")
        .legend( |(x,y)| Path::new(vec![(x,y), (x + 20,y)], &RED) );

    chart2.configure_series_labels()
        .background_style( &WHITE.mix(0.8) )
        .border_style( &BLACK )
        .draw()?;

    Ok(())
}

fn main() {
    
    let kuhnpoker = KuhnPoker::new(1e-3, 2e-1);
    let algo = PolicyGradientAlgorithm::ProximalPolicyOptimisation;
    let (loss_critic,loss_actor) = kuhnpoker.train(&algo, 1000, 100);
    plot(loss_critic,loss_actor).expect("Plot function failed");

}