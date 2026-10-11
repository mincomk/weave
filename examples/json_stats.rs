//! Headless: load a JSON graph, seed it with the tidy layout, and run `QuadTreeUpdater` ticks —
//! reporting the runtime and how the layout behaves. No window, so it runs over SSH/CI and is the
//! way to check a `--depth` slice, or a set of updater constants, without watching a screen.
//!
//! ```sh
//! cargo run -r --example json_stats -- follow-graph-weave.json --depth 2 --ticks 300
//! cargo run -r --example json_stats -- follow-graph-weave.json --depth 2 --ticks 300 --soft
//! ```

use clap::Parser;
use glam::Vec2;
use petgraph::visit::EdgeRef;
use petgraph::{
    Directed, Direction,
    stable_graph::{NodeIndex, StableGraph},
};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::Instant;
use weave::json_loader::GraphData;
use weave::layout::seed_layout;
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
    /// How many updater ticks to run.
    #[arg(long, default_value_t = 120)]
    ticks: usize,
    /// Print a line every N ticks.
    #[arg(long, default_value_t = 20)]
    every: usize,
    /// Use `QuadTreeUpdater::default_setting()` instead of the tuned constants.
    #[arg(long, default_value_t = false)]
    defaults: bool,
}

struct Body {
    position: Vec2,
    prev_position: Vec2,
}

/// Resident set size in MB, from /proc — so a runaway tree is visible while it is happening.
fn rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| {
            s.split_whitespace()
                .nth(1)
                .and_then(|pages| pages.parse::<f64>().ok())
        })
        .map(|pages| pages * 4096.0 / 1e6)
        .unwrap_or(0.0)
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

    let (positions, _depths) = seed_layout(&data);

    println!("file       : {}", args.file.display());
    println!("full graph : {} nodes, {} edges (before the depth filter)", full.0, full.1);
    println!("loaded     : {nodes} nodes, {edges} edges");
    if args.depth != u32::MAX {
        println!("depth      : <= {}", args.depth);
    }

    let mut graph = StableGraph::<Body, (), Directed>::new();
    let mut id_to_index: HashMap<i64, NodeIndex> = HashMap::with_capacity(nodes);
    for (i, node) in data.nodes.iter().enumerate() {
        let position = positions[i];
        id_to_index.insert(
            node.id,
            graph.add_node(Body {
                position,
                prev_position: position,
            }),
        );
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

    let updater = if args.defaults {
        QuadTreeUpdater::default_setting()
    } else {
        QuadTreeUpdater::unit_scale_setting()
    };
    println!(
        "updater    : {}",
        if args.defaults {
            "default_setting()"
        } else {
            "unit_scale_setting()"
        }
    );

    let degrees: Vec<usize> = graph
        .node_indices()
        .map(|i| graph.edges_directed(i, Direction::Incoming).count())
        .collect();

    println!("\n tick   ms/tick      rmax    median       p99  rss MB   farthest (in-degree)");
    for tick in 0..=args.ticks {
        if tick > 0 {
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

            if tick % args.every == 0 || tick == args.ticks {
                let mut dist: Vec<(f32, NodeIndex)> = graph
                    .node_indices()
                    .map(|i| (graph[i].position.length(), i))
                    .collect();
                dist.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

                let rmax = dist.first().map(|d| d.0).unwrap_or(0.0);
                let median = dist.get(dist.len() / 2).map(|d| d.0).unwrap_or(0.0);
                let p99 = dist
                    .get(dist.len() * 99 / 100)
                    .map(|d| d.0)
                    .unwrap_or(0.0);
                let top: Vec<String> = dist
                    .iter()
                    .take(3)
                    .map(|(d, i)| format!("{d:.0}({})", degrees[i.index()]))
                    .collect();

                println!(
                    "{tick:5} {:8.2} {:9.1} {:8.1} {:9.1} {:6.0}   {}",
                    elapsed.as_secs_f64() * 1e3,
                    rmax,
                    median,
                    p99,
                    rss_mb(),
                    top.join(" ")
                );
            }
        }
    }
}
