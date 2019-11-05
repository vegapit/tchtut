extern crate tch;

use tch::{nn, Tensor, Kind, nn::OptimizerConfig};

fn norm_cdf(x: &Tensor) -> Tensor {
    0.5 * ( 1.0 + ( x / Tensor::from(2.0).sqrt() ).erf() )
}

fn black76(epsilon: &Tensor, f: &Tensor, k: &Tensor, t: &Tensor, sigma: &Tensor, r: &Tensor) -> Tensor {
    let d1 = ((f/k).log() + Tensor::from(0.5) * sigma.pow(2.0) * t) / ( sigma * t.sqrt() );
    let d2 = &d1 - sigma * t.sqrt();
    epsilon * (-r * t).exp() * ( f * norm_cdf(&(epsilon * d1)) - k * norm_cdf(&(epsilon * d2)) )
}

fn func_builder(p: nn::Path) -> impl Fn(&Tensor,&Tensor,&Tensor,&Tensor,&Tensor) -> Tensor {
    let sigma = p.randn_standard("sigma", &[1]);
    move |epsilon, f, k , t ,r| {
        black76(&epsilon, &f, &k, &t, &sigma, &r)
    }
}

fn main() {
    tch::manual_seed(0);
    
    let vs = nn::VarStore::new( tch::Device::Cpu );
    let black76_volsolver = func_builder( vs.root() );
    let mut opt = nn::Adam::default().build(&vs, 1e-2).unwrap();

    let epsilon = Tensor::from(1f64);
    let f = Tensor::from(100f64);
    let k = Tensor::from(100f64);
    let t = Tensor::from(1f64);
    let r = Tensor::from(0.01);
    let price = Tensor::from(11.805);

    loop {
        let square_loss = (black76_volsolver(&epsilon, &f, &k, &t, &r) - &price).pow(2f64).sum( Kind::Float );
        opt.backward_step(&square_loss);
        println!("{}", f64::from(&square_loss) );
        if f64::from(&square_loss) < 0.0001 {
            break;
        }
    }

    let sigma = &vs.root().get("sigma").unwrap();
    let calc_price = f64::from( black76(&epsilon, &f, &k, &t, &sigma, &r) );
    assert!( (calc_price - f64::from(price)).abs() < 0.01  );
}