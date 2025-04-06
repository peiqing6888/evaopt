use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use evaopt_core::matrix::{MatrixConfig, DecompositionMethod, optimize_matrix};
use ndarray::Array2;
use rand::Rng;

fn create_test_matrix(size: usize, sparsity: f32) -> Array2<f32> {
    let mut rng = rand::thread_rng();
    Array2::from_shape_fn((size, size), |_| {
        if rng.gen::<f32>() > sparsity {
            rng.gen::<f32>() * 2.0 - 1.0
        } else {
            0.0
        }
    })
}

fn bench_svd_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("svd_optimization");
    for size in [128, 256, 512].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let matrix = create_test_matrix(size, 0.5);
            let config = MatrixConfig {
                method: DecompositionMethod::SVD,
                rank: size / 4,
                tolerance: 1e-6,
                use_parallel: true,
                oversampling: 5,
                power_iterations: 2,
                block_sparse: None,
            };
            
            b.iter(|| {
                black_box(optimize_matrix(&matrix, &config).unwrap());
            });
        });
    }
    group.finish();
}

fn bench_low_rank_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("low_rank_optimization");
    for size in [128, 256, 512].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let matrix = create_test_matrix(size, 0.5);
            let config = MatrixConfig {
                method: DecompositionMethod::LowRank,
                rank: size / 4,
                tolerance: 1e-6,
                use_parallel: true,
                oversampling: 5,
                power_iterations: 2,
                block_sparse: None,
            };
            
            b.iter(|| {
                black_box(optimize_matrix(&matrix, &config).unwrap());
            });
        });
    }
    group.finish();
}

fn bench_sparse_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_optimization");
    for size in [128, 256, 512].iter() {
        for sparsity in [0.5, 0.7, 0.9].iter() {
            group.bench_with_input(
                BenchmarkId::new("size_sparsity", format!("{}_{}", size, sparsity)),
                &(*size, *sparsity),
                |b, &(size, sparsity)| {
                    let matrix = create_test_matrix(size, sparsity);
                    let config = MatrixConfig {
                        method: DecompositionMethod::Sparse,
                        rank: size / 4,
                        tolerance: 1e-6,
                        use_parallel: true,
                        oversampling: 5,
                        power_iterations: 2,
                        block_sparse: None,
                    };
                    
                    b.iter(|| {
                        black_box(optimize_matrix(&matrix, &config).unwrap());
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_svd_optimization,
    bench_low_rank_optimization,
    bench_sparse_optimization
);
criterion_main!(benches); 