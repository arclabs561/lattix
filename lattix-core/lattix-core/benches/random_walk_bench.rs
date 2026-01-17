use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lattix_core::algo::random_walk::{generate_walks, RandomWalkConfig};
use lattix_core::{KnowledgeGraph, Triple};

fn bench_random_walks(c: &mut Criterion) {
    let mut kg = KnowledgeGraph::new();
    // Create a simple connected graph: ring of 1000 nodes
    for i in 0..1000 {
        let s = format!("node_{}", i);
        let o = format!("node_{}", (i + 1) % 1000);
        kg.add_triple(Triple::new(s.as_str(), "connects_to", o.as_str()));
    }

    let config = RandomWalkConfig {
        walk_length: 10,
        num_walks: 5,
        p: 1.0,
        q: 1.0,
        seed: 42,
    };

    c.bench_function("random_walk_1000_nodes", |b| {
        b.iter(|| generate_walks(black_box(&kg), black_box(config)))
    });
}

criterion_group!(benches, bench_random_walks);
criterion_main!(benches);
