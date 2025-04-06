use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use evaopt_core::{Model, Layer, dynamic::OptimizerConfig};
use ndarray::{Array2, Array1};

fn create_test_model(size: usize) -> Model {
    let config = OptimizerConfig {
        activation_threshold: 0.1,
        min_active_ratio: 0.5,
        update_frequency: 100,
        ema_alpha: 0.1,
        cache_enabled: true,
    };
    
    let mut model = Model::with_optimizer(config);
    
    // Add test layers with corrected dimensions
    let layer1 = Layer::new(
        "fc1".to_string(),
        Array2::from_shape_fn((size, size), |_| rand::random::<f32>() * 2.0 - 1.0),
        Some(Array1::zeros(size)),
    ).unwrap();
    
    let layer2 = Layer::new(
        "fc2".to_string(),
        Array2::from_shape_fn((size/2, size), |_| rand::random::<f32>() * 2.0 - 1.0),
        Some(Array1::zeros(size/2)),
    ).unwrap();
    
    model.add_layer(layer1);
    model.add_layer(layer2);
    
    model
}

fn bench_forward_pass(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_pass");
    for size in [512, 1024, 2048].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let model = create_test_model(size);
            let input = Array2::from_shape_fn((32, size), |_| rand::random::<f32>() * 2.0 - 1.0);
            
            b.iter(|| {
                black_box(model.forward(input.clone()).unwrap());
            });
        });
    }
    group.finish();
}

fn bench_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization");
    for size in [512, 1024, 2048].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let model = create_test_model(size);
            let input = Array2::from_shape_fn((32, size), |_| rand::random::<f32>() * 2.0 - 1.0);
            
            // Run forward passes to collect statistics
            for _ in 0..100 {
                black_box(model.forward(input.clone()).unwrap());
            }
            
            // Verify that statistics are collected
            for layer_name in ["fc1", "fc2"] {
                black_box(model.get_layer_stats(layer_name).unwrap());
            }
            
            b.iter(|| {
                black_box(model.optimize().unwrap());
            });
        });
    }
    group.finish();
}

fn bench_activation_tracking(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_tracking");
    for size in [512, 1024, 2048].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let model = create_test_model(size);
            let input = Array2::from_shape_fn((32, size), |_| rand::random::<f32>() * 2.0 - 1.0);
            let _output = model.forward(input.clone()).unwrap();
            
            b.iter(|| {
                for layer_name in ["fc1", "fc2"] {
                    black_box(model.get_layer_stats(layer_name).unwrap());
                }
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_forward_pass,
    bench_optimization,
    bench_activation_tracking
);
criterion_main!(benches); 