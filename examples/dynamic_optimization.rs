use evaopt_core::{Model, Layer, OptimizerConfig, LayerStats};
use ndarray::{Array2, Array1};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a model with dynamic optimization
    let config = OptimizerConfig {
        activation_threshold: 0.1,
        min_active_ratio: 0.5,  // We want to reduce to 50% neurons
        update_frequency: 100,
        ema_alpha: 0.1,
        cache_enabled: true,
    };
    
    let mut model = Model::with_optimizer(config);
    
    // Add some example layers
    let layer1 = Layer::new(
        "fc1".to_string(),
        Array2::from_shape_fn((1024, 512), |_| rand::random::<f32>() * 2.0 - 1.0),
        Some(Array1::zeros(1024)),
    )?;
    
    let layer2 = Layer::new(
        "fc2".to_string(),
        Array2::from_shape_fn((512, 256), |_| rand::random::<f32>() * 2.0 - 1.0),
        Some(Array1::zeros(512)),
    )?;
    
    let layer3 = Layer::new(
        "fc3".to_string(),
        Array2::from_shape_fn((256, 128), |_| rand::random::<f32>() * 2.0 - 1.0),
        Some(Array1::zeros(256)),
    )?;
    
    model.add_layer(layer1);
    model.add_layer(layer2);
    model.add_layer(layer3);
    
    // Run some forward passes to collect activation statistics
    println!("Running forward passes to collect statistics...");
    for i in 0..1000 {
        let input = Array2::from_shape_fn((32, 512), |_| rand::random::<f32>() * 2.0 - 1.0);
        let _ = model.forward(input)?;
        
        if i % 100 == 0 {
            // Print statistics for each layer
            for layer_name in ["fc1", "fc2", "fc3"] {
                let stats = model.get_layer_stats(layer_name)?;
                println!(
                    "Layer {}: {}/{} neurons active ({:.1}% compression), {} frozen",
                    layer_name,
                    stats.active_neurons,
                    stats.total_neurons,
                    stats.compression_ratio * 100.0,
                    stats.frozen_neurons
                );
            }
            println!();
        }
    }
    
    // Optimize the model
    println!("\nOptimizing model...");
    let optimized = model.optimize()?;
    
    // Print final statistics
    for layer_name in ["fc1", "fc2", "fc3"] {
        let stats = optimized.get_layer_stats(layer_name)?;
        println!(
            "Layer {}: Final size {}/{} neurons ({:.1}% reduction)",
            layer_name,
            stats.active_neurons,
            stats.total_neurons,
            stats.compression_ratio * 100.0
        );
    }
    
    Ok(())
} 