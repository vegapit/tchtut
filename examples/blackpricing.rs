extern crate tch;

use tch::Tensor;

fn norm_cdf(x: &Tensor) -> Tensor {
    0.5 * ( 1.0 + ( x / Tensor::from(2.0).sqrt() ).erf() )
}

fn black76(epsilon: &Tensor, f: &Tensor, k: &Tensor, t: &Tensor, sigma: &Tensor, r: &Tensor) -> Tensor {
    let d1 = ((f/k).log() + Tensor::from(0.5) * sigma.pow(2.0) * t) / ( sigma * t.sqrt() );
    let d2 = &d1 - sigma * t.sqrt();
    epsilon * (-r * t).exp() * ( f * norm_cdf(&(epsilon * d1)) - k * norm_cdf(&(epsilon * d2)) )
}

fn main() {
    
    let epsilon = Tensor::from(1f64);
    let f = Tensor::from(100f64).set_requires_grad(true);
    let k = Tensor::from(100f64);
    let sigma = Tensor::from(0.3).set_requires_grad(true);
    let t = Tensor::from(1f64);
    let r = Tensor::from(0.01);

    let price = &black76(&epsilon, &f, &k, &t, &sigma, &r);
   
    let price_grad = Tensor::run_backward(&[price], &[&f,&sigma], true, true);
    let delta = &price_grad[0];
    let vega = &price_grad[1];

    let delta_grad = Tensor::run_backward(&[delta], &[&f], true, false);
    let gamma = &delta_grad[0];

    let vega_grad = Tensor::run_backward(&[vega], &[&f,&sigma], false, false);
    let vanna = &vega_grad[0];
    let volga = &vega_grad[1];

    println!("Price: {:.4}", f64::from(price));
    println!("Delta: {:.4}", f64::from(delta));
    println!("Gamma: {:.4}", f64::from(gamma));
    println!("Vega: {:.4}", f64::from(vega));
    println!("Vanna: {:.4}", f64::from(vanna));
    println!("Volga: {:.4}", f64::from(volga));
}