use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use evaopt_core::matrix::{MatrixConfig, DecompositionMethod, optimize_matrix, BlockSparseConfig};
use ndarray::Array2;
use rand::Rng;

fn create_block_matrix(size: usize, block_size: usize) -> Array2<f32> {
    let mut rng = rand::thread_rng();
    Array2::from_shape_fn((size, size), |(i, j)| {
        let block_i = i / block_size;
        let block_j = j / block_size;
        if block_i == block_j {
            rng.gen::<f32>() * 2.0 - 1.0
        } else {
            0.0
        }
    })
}

fn bench_block_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_optimization");
    for size in [128, 256, 512].iter() {
        for block_size in [16, 32, 64].iter() {
            if *block_size >= *size { continue; }
            
            group.bench_with_input(
                BenchmarkId::new("size_block", format!("{}_{}", size, block_size)),
                &(*size, *block_size),
                |b, &(size, block_size)| {
                    let matrix = create_block_matrix(size, block_size);
                    let config = MatrixConfig {
                        method: DecompositionMethod::BlockSparse,
                        rank: block_size,
                        tolerance: 1e-6,
                        use_parallel: true,
                        oversampling: 5,
                        power_iterations: 2,
                        block_sparse: Some(BlockSparseConfig {
                            block_size,
                            sparsity_threshold: 0.1,
                            min_block_norm: 0.01,
                        }),
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

criterion_group!(benches, bench_block_optimization);
criterion_main!(benches); 