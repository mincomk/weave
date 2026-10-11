//! Headless: load a JSON graph, build the same `PhysicalGraph` the visual example builds, and run
//! one `QuadTreeUpdater` tick — printing sizes and timing. No window, so it runs over SSH/CI and
//! is the quick way to check that a `--depth` slice is small enough to simulate.
//!
//! ```sh
//! cargo run -r --example json_stats -- follow-graph-weave.json --depth 2
//! ```

use clap::Parser;
use glam::Vec2;
use petgraph::{
    Directed, Direction,
    stable_graph::{NodeIndex, StableGraph},
};
use petgraph::visit::EdgeRef;
use rand::Rng;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::Instant;
use weave::json_loader::GraphData;
use weave::updater::{QuadTreeUpdater, Updater};
use weave::{PhysicalGraph, PhysicalNode};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the JSON graph file.
    file: PathBuf,
    /// Load only nodes with depth <= this (0 = seed). Default: everything.
    #[arg(short, long, default_value_t = u32::MAX)]
    depth: u32,
    /// How many updater ticks to time.
    #[arg(long, default_value_t = 1)]
    ticks: usize,
}

struct Body {
    position: Vec2,
    prev_position: Vec2,
}

impl Body {
    fn random(radius: f32) -> Self {
        let mut rng = rand::rng();
        let position = Vec2::new(
            rng.random_range(-radius..radius),
            rng.random_range(-radius..radius),
        );
        Body {
            position,
            prev_position: position,
        }
    }
}

fn main() {
    let args = Args::parse();

    let data = match GraphData::from_path(&args.file) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("failed to load {}: {e}", args.file.display());
            std::process::exit(1);
        }
    };
    let full = data.counts();
    let data = if args.depth == u32::MAX {
        data
    } else {
        data.filter_depth(args.depth)
    };
    let (nodes, edges) = data.counts();

    println!(
        "file       : {}",
        args.file.display()
    );
    println!("full graph : {} nodes, {} edges (before depth filter)", full.0, full.1);
    println!("loaded     : {nodes} nodes, {edges} edges");
    if args.depth != u32::MAX {
        println!("depth      : <= {}", args.depth);
    }

    let mut graph = StableGraph::<Body, (), Directed>::new();
    let mut id_to_index: HashMap<i64, NodeIndex> = HashMap::with_capacity(nodes);
    for node in &data.nodes {
        id_to_index.insert(node.id, graph.add_node(Body::random(1.0)));
    }
    let mut dropped_edges = 0usize;
    for edge in &data.edges {
        match (id_to_index.get(&edge.source), id_to_index.get(&edge.target)) {
            (Some(&s), Some(&t)) if s != t => {
                graph.add_edge(s, t, ());
            }
            _ => dropped_edges += 1,
        }
    }

    if dropped_edges > 0 {
        println!("skipped    : {dropped_edges} edges (dangling or self-loop)");
    }
    if nodes > 5_000 {
        println!(
            "\nwarning: {nodes} nodes is a lot for the current O(n^2) neighbour pass; try a smaller --depth."
        );
    }

    let updater = QuadTreeUpdater::default_setting();
    let mut last = Instant::now();

    for tick in 0..args.ticks {
        let physical_graph = PhysicalGraph::new(
            graph
                .node_indices()
                .map(|node_index| {
                    let node = PhysicalNode::new(
                        node_index,
                        graph
                            .edges_directed(node_index, Direction::Incoming)
                            .map(|e| e.source())
                            .collect::<HashSet<NodeIndex>>(),
                        graph
                            .edges_directed(node_index, Direction::Outgoing)
                            .map(|e| e.target())
                            .collect::<HashSet<NodeIndex>>(),
                        graph[node_index].position,
                        graph[node_index].prev_position,
                    );
                    (node_index, node)
                })
                .collect::<HashMap<NodeIndex, PhysicalNode<NodeIndex>>>(),
        );

        let started = Instant::now();
        let updated = updater.update(&physical_graph, 1.0 / 60.0);
        let elapsed = started.elapsed();

        for (node_index, node) in updated.nodes {
            graph[node_index].position = node.position;
            graph[node_index].prev_position = node.prev_position;
        }

        println!(
            "tick {tick}: {:.1} ms  (mass sum = {:.1})",
            elapsed.as_secs_f64() * 1e3,
            graph
                .node_indices()
                .map(|i| updater.min_node_mass
                    + graph.edges_directed(i, Direction::Incoming).count() as f32
                        * updater.mass_amplification)
                .sum::<f32>()
        );
        last = Instant::now();
    }
    let _ = last;
}
