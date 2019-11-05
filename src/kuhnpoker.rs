#![allow(non_snake_case)]
use indicatif::ProgressBar;
use tch::{Tensor, nn, nn::Module, nn::OptimizerConfig, Kind, Device, IndexOp};

pub enum PolicyGradientAlgorithm {
    VanillaPolicyGradient,
    ProximalPolicyOptimisation
}

// Helper functions 

fn showdown(cards: &[i64;2], pot: f32) -> f32 {
    if cards[0] > cards[1] {
        pot
    } else if cards[0] < cards[1] {
        -pot
    } else {
        panic!("Both players can not have the same card")
    }
}

fn payoff(cards: &[i64;2], actionset: &[i64;4]) -> [f32;4] {
    match &actionset {
        [0,_,1,0] => [-1f32,0f32,1f32,-1f32], // Check + Bet + Fold
        [0,_,1,1] => { // Check + Bet + Call
            let payout = showdown(cards, 2f32);
            [payout,0f32,-payout,payout]
        }, 
        [0,_,0,_] => { // Check + Check
            let payout = showdown(cards, 1f32);
            [payout,0f32,-payout,0f32]
        }, 
        [1,0,_,_] => [1f32,-1f32,0f32,0f32], // Bet + Fold
        [1,1,_,_] => { // Bet + Call
            let payout = showdown(cards, 2f32);
            [payout,-payout,0f32,0f32]
        },
        _ => panic!("Actions not recognized")
    }
}

pub struct KuhnPoker{
    pub lr: f64,
    pub epsilon: f64
}

impl KuhnPoker {

    pub fn new(lr: f64, epsilon: f64) -> Self {
        Self {
            lr: lr,
            epsilon: epsilon
        }
    }

    // Actors
    fn actor1A_builder(p: nn::Path) -> nn::Sequential {
        nn::seq()
            .add( nn::linear(&p / "actor_layer1_1A", 1, 7, Default::default()) )
            .add_fn(|xs| xs.relu())
            .add( nn::linear(&p / "actor_layer2_1A", 7, 2, Default::default()) )
            .add_fn(|xs| xs.softmax(1, Kind::Float))
    }

    fn actor2A_builder(p: nn::Path) -> nn::Sequential {
        nn::seq()
            .add( nn::linear(&p / "actor_layer1_2A", 1, 7, Default::default()) )
            .add_fn(|xs| xs.relu())
            .add( nn::linear(&p / "actor_layer2_2A", 7, 2, Default::default()) )
            .add_fn(|xs| xs.softmax(1, Kind::Float))
    }

    fn actor2B_builder(p: nn::Path) -> nn::Sequential {
        nn::seq()
            .add( nn::linear(&p / "actor_layer1_2B", 1, 7, Default::default()) )
            .add_fn(|xs| xs.relu())
            .add( nn::linear(&p / "actor_layer2_2B", 7, 2, Default::default()) )
            .add_fn(|xs| xs.softmax(1, Kind::Float))
    }

    fn actor1B_builder(p: nn::Path) -> nn::Sequential {
        nn::seq()
            .add( nn::linear(&p / "actor_layer1_1B", 1, 7, Default::default()) )
            .add_fn(|xs| xs.relu())
            .add( nn::linear(&p / "actor_layer2_1B", 7, 2, Default::default()) )
            .add_fn(|xs| xs.softmax(1, Kind::Float))
    }

    // Critics
    fn critic1A_builder(p: nn::Path) -> nn::Sequential {
        nn::seq()
            .add( nn::linear(&p / "critic_layer1_1A", 1, 7, Default::default()) )
            .add_fn(|xs| xs.relu())
            .add( nn::linear(&p / "critic_layer2_1A", 7, 1, Default::default()) )
    }

    fn critic2A_builder(p: nn::Path) -> nn::Sequential {
        nn::seq()
            .add( nn::linear(&p / "critic_layer1_2A", 1, 7, Default::default()) )
            .add_fn(|xs| xs.relu())
            .add( nn::linear(&p / "critic_layer2_2A", 7, 1, Default::default()) )
    }

    fn critic2B_builder(p: nn::Path) -> nn::Sequential {
        nn::seq()
            .add( nn::linear(&p / "critic_layer1_2B", 1, 7, Default::default()) )
            .add_fn(|xs| xs.relu())
            .add( nn::linear(&p / "critic_layer2_2B", 7, 1, Default::default()) )
    }

    fn critic1B_builder(p: nn::Path) -> nn::Sequential {
        nn::seq()
            .add( nn::linear(&p / "critic_layer1_1B", 1, 7, Default::default()) )
            .add_fn(|xs| xs.relu())
            .add( nn::linear(&p / "critic_layer2_1B", 7, 1, Default::default()) )
    }

    // PPO helper
    fn ppo_g(&self, a: &Tensor) -> Tensor {
        a.zeros_like().max1( &(&Tensor::from( 1f32 + (self.epsilon as f32) ) * a ) ) + a.zeros_like().min1( &(&Tensor::from( 1f32 - (self.epsilon as f32) ) * a ) )
    }

    pub fn train(&self, algo: &PolicyGradientAlgorithm, epoch_num: usize, min_batch_size: usize) -> (Vec<(f32,f32)>,Vec<(f32,f32)>) {

        tch::manual_seed(0);
        
        let vs_actor1A = nn::VarStore::new( tch::Device::Cpu );
        let vs_actor2A = nn::VarStore::new( tch::Device::Cpu );
        let vs_actor2B = nn::VarStore::new( tch::Device::Cpu );
        let vs_actor1B = nn::VarStore::new( tch::Device::Cpu );
        
        let vs_critic1A = nn::VarStore::new( tch::Device::Cpu );
        let vs_critic2A = nn::VarStore::new( tch::Device::Cpu );
        let vs_critic2B = nn::VarStore::new( tch::Device::Cpu );
        let vs_critic1B = nn::VarStore::new( tch::Device::Cpu );
        
        let actor1A = KuhnPoker::actor1A_builder( vs_actor1A.root() );
        let actor2A = KuhnPoker::actor2A_builder( vs_actor2A.root() );
        let actor2B = KuhnPoker::actor2B_builder( vs_actor2B.root() );
        let actor1B = KuhnPoker::actor1B_builder( vs_actor1B.root() );
        
        let critic1A = KuhnPoker::critic1A_builder( vs_critic1A.root() );
        let critic2A = KuhnPoker::critic2A_builder( vs_critic2A.root() );
        let critic2B = KuhnPoker::critic2B_builder( vs_critic2B.root() );
        let critic1B = KuhnPoker::critic1B_builder( vs_critic1B.root() );
        
        let mut opt_actor1A = nn::Adam::default().build( &vs_actor1A, self.lr ).unwrap();
        let mut opt_actor2A = nn::Adam::default().build( &vs_actor2A, self.lr ).unwrap();
        let mut opt_actor2B = nn::Adam::default().build( &vs_actor2B, self.lr ).unwrap();
        let mut opt_actor1B = nn::Adam::default().build( &vs_actor1B, self.lr ).unwrap();
        
        let mut opt_critic1A = nn::Adam::default().build( &vs_critic1A, self.lr ).unwrap();
        let mut opt_critic2A = nn::Adam::default().build( &vs_critic2A, self.lr ).unwrap();
        let mut opt_critic2B = nn::Adam::default().build( &vs_critic2B, self.lr ).unwrap();
        let mut opt_critic1B = nn::Adam::default().build( &vs_critic1B, self.lr ).unwrap();
        
        let mut loss_actor_data : Vec<(f32,f32)> = Vec::new();
        let mut loss_critic_data : Vec<(f32,f32)> = Vec::new();
        let mut avg_loss_critics = 0f32;
        let mut avg_loss_actors = 0f32;

        let bar = ProgressBar::new(epoch_num as u64);

        let actors = [ &actor1A, &actor2A, &actor2B, &actor1B ];
        let critics = [ &critic1A, &critic2A, &critic2B, &critic1B ];

        for i in 0..epoch_num {

            let mut reward_sets : [Vec<Tensor>; 4] = [Vec::new(),Vec::new(),Vec::new(),Vec::new()];
            let mut action_sets : [Vec<Tensor>; 4] = [Vec::new(),Vec::new(),Vec::new(),Vec::new()];
            let mut state_sets : [Vec<Tensor>; 4] = [Vec::new(),Vec::new(),Vec::new(),Vec::new()];
            let mut baseline_sets : [Vec<Tensor>; 4] = [Vec::new(),Vec::new(),Vec::new(),Vec::new()];
            let mut probs_sets : [Vec<Tensor>; 4] = [Vec::new(),Vec::new(),Vec::new(),Vec::new()];

            let mut counter = 0;
            loop {

                let shuffle = Tensor::of_slice(&[1f32,1f32,1f32]).multinomial(2, false).to_kind(Kind::Float);

                let mut cards : [i64; 2] = [0, 0];
                cards.copy_from_slice( Vec::<i64>::from(&shuffle).as_slice() );
                
                let state_set = [ cards[0] - 1 , cards[1] - 1, cards[1] - 1, cards[0] - 1 ];

                tch::no_grad( || {
                    let probs1A = actor1A.forward( &Tensor::of_slice(&[state_set[0]]).to_kind(Kind::Float).unsqueeze(0) );
                    let probs2A = actor2A.forward( &Tensor::of_slice(&[state_set[1]]).to_kind(Kind::Float).unsqueeze(0) );
                    let probs2B = actor2B.forward( &Tensor::of_slice(&[state_set[2]]).to_kind(Kind::Float).unsqueeze(0) );
                    let probs1B = actor1B.forward( &Tensor::of_slice(&[state_set[3]]).to_kind(Kind::Float).unsqueeze(0) );

                    let action1A = probs1A.multinomial(1, false);
                    let action2A = probs2A.multinomial(1, false);
                    let action2B = probs2B.multinomial(1, false);
                    let action1B = probs1B.multinomial(1, false);

                    let actions = [ i64::from(&action1A), i64::from(&action2A), i64::from(&action2B), i64::from(&action1B) ];
                    let probs = [ Vec::<f32>::from( probs1A.unsqueeze(0) ), Vec::<f32>::from( probs2A.unsqueeze(0) ), Vec::<f32>::from( probs2B.unsqueeze(0) ), Vec::<f32>::from( probs1B.unsqueeze(0) ) ];
                    let rewards = payoff(&cards, &actions);

                    for j in 0..4 {
                        if rewards[j] != 0f32 { // If reward is zero, the player's action is not relevant to the payoff so ignore it
                            let baseline = critics[j].forward( &Tensor::of_slice( &[ state_set[j] ] ).to_kind(Kind::Float).unsqueeze(0) );
                            baseline_sets[j].push( Tensor::of_slice( &[ f32::from(baseline) ] ) );
                            probs_sets[j].push( Tensor::of_slice( &[ probs[j][ actions[j] as usize ] ] ) );
                            reward_sets[j].push( Tensor::of_slice( &[ rewards[j] ] ) );
                            action_sets[j].push( Tensor::of_slice( &[ actions[j] ] ) );
                            state_sets[j].push( Tensor::of_slice( &[ state_set[j] ] ) );
                            if j == 3 { // Counts how many times 1B has played
                                counter += 1;
                            }
                        }
                    }
                });

                if counter == min_batch_size { // If player1B has played enough times, exit loop
                    break;
                }
            }

            let mut loss_actors = 0f32;
            let mut loss_critics = 0f32;
            
            for k in 0..4 {

                let sample_size = reward_sets[k].len() as i64;

                if sample_size > 0 {
                    let rewards_t = Tensor::stack( reward_sets[k].as_slice(), 0 );
                    let actions_t = Tensor::stack( action_sets[k].as_slice(), 0 );
                    let states_t = Tensor::stack( state_sets[k].as_slice(), 0 ).to_kind(Kind::Float);
                    let baselines_t = Tensor::stack( baseline_sets[k].as_slice(), 0 );
                    let probs_t = Tensor::stack( probs_sets[k].as_slice(), 0 );

                    let action_mask = Tensor::zeros( &[sample_size,2], (Kind::Float,Device::Cpu) ).scatter1( 1, &actions_t, 1.0 );
                    
                    let probs = ( actors[k].forward(&states_t) * &action_mask ).sum1(&[1], false, Kind::Float);
                    let advantages = &rewards_t - baselines_t;
                    let loss_actor = match algo {
                        PolicyGradientAlgorithm::ProximalPolicyOptimisation => -self.ppo_g( &advantages ).min1( &(&advantages * &probs.unsqueeze(1) / &probs_t) ).mean( Kind::Float ),
                        PolicyGradientAlgorithm::VanillaPolicyGradient => -( &advantages * &probs.unsqueeze(1) ).mean( Kind::Float )
                    };

                    match k {
                        0 => opt_actor1A.backward_step( &loss_actor ),
                        1 => opt_actor2A.backward_step( &loss_actor ),
                        2 => opt_actor2B.backward_step( &loss_actor ),
                        3 => opt_actor1B.backward_step( &loss_actor ),
                        _ => panic!("Calculation incoherence")
                    }
                    
                    loss_actors += f32::from(&loss_actor) * 0.25;
                    
                    let values = critics[k].forward( &states_t );
                    let loss_critic = ( &rewards_t - values ).pow(2f64).mean( Kind::Float );
                    
                    match k {
                        0 => opt_critic1A.backward_step( &loss_critic ),
                        1 => opt_critic2A.backward_step( &loss_critic ),
                        2 => opt_critic2B.backward_step( &loss_critic ),
                        3 => opt_critic1B.backward_step( &loss_critic ),
                        _ => panic!("Calculation incoherence")
                    }

                    loss_critics += f32::from(&loss_critic) * 0.25;
                }

            }

            if avg_loss_critics == 0f32 {
                avg_loss_critics = loss_critics;
            } else {
                avg_loss_critics += 0.2 * (loss_critics - avg_loss_critics);
            }
            
            if avg_loss_actors == 0f32 {
                avg_loss_actors = loss_actors;
            } else {
                avg_loss_actors += 0.2 * (loss_actors - avg_loss_actors);
            }

            loss_critic_data.push( (i as f32, avg_loss_critics) );
            loss_actor_data.push( (i as f32, avg_loss_actors) );
            
            bar.inc(1);
        }
        
        bar.finish();

        let states_sample = &Tensor::of_slice(&[-1f32, 0f32, 1f32]).view([3,1]);

        fn format_policy(v: &Tensor) {
            println!( "J : {:.2}% | {:.2}%", 100.0 * f64::from( v.i((0,0)) ), 100.0 * f64::from( v.i((0,1)) ) );
            println!( "Q : {:.2}% | {:.2}%", 100.0 * f64::from( v.i((1,0)) ), 100.0 * f64::from( v.i((1,1)) ) );
            println!( "K : {:.2}% | {:.2}%", 100.0 * f64::from( v.i((2,0)) ), 100.0 * f64::from( v.i((2,1)) ) );
        }

        println!("Policy 1A (CHECK | BET)");
        let v1A = actor1A.forward( &states_sample );
        format_policy( &v1A );

        println!("Policy 1B (FOLD | CALL)");
        let v1B = actor1B.forward( &states_sample );
        format_policy( &v1B );
        
        println!("Policy 2A (FOLD | CALL)");
        let v2A = actor2A.forward( &states_sample );
        format_policy( &v2A );
        
        println!("Policy 2B (CHECK | BET)");
        let v2B = actor2B.forward( &states_sample );
        format_policy( &v2B );
        
        let values1A = Vec::<f64>::from( critic1A.forward( &states_sample ).view([3,1]) );
        println!("Player1 edge: {:.4}", &values1A.iter().sum::<f64>() / 3f64 );

        (loss_critic_data,loss_actor_data)
    }

}