//! Interactive visualiser for a JSON graph, driven by `QuadTreeUpdater` and `scrollrs`.
//!
//! * left-drag  — pan
//! * wheel      — zoom
//! * space      — start / stop the physics
//! * `f`        — re-fit the view to the whole graph
//!
//! ```sh
//! # small slice first
//! cargo run -r --example json_graph -- follow-graph-weave.json -w 0.3 --depth 2
//! # to see the whole thing (slow: the neighbour pass is O(n^2))
//! cargo run -r --example json_graph -- follow-graph-weave.json -w 0.3 --depth 3
//! ```
//!
//! The scroll/camera handling is the same as `node_physics` / `file_tree`; what differs is the
//! graph source (a JSON file) and sane init/fit for a graph that can be thousands of nodes.

use clap::Parser;
use glam::{DVec2, Vec2};
use macroquad::{
    color,
    input::{
        KeyCode, MouseButton, is_key_pressed, is_mouse_button_down, mouse_position, mouse_wheel,
    },
    shapes::{draw_circle, draw_line},
    text::draw_text,
    time::{get_fps, get_frame_time},
    window::{Conf, clear_background, next_frame, screen_height, screen_width},
};
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use petgraph::{
    Directed, Direction,
    stable_graph::{NodeIndex, StableGraph},
};
use rand::Rng;
use scrollrs::Projector;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use weave::json_loader::GraphData;
use weave::updater::{QuadTreeUpdater, Updater};
use weave::{PhysicalGraph, PhysicalNode};

fn window_conf() -> Conf {
    Conf {
        window_title: "weave — JSON graph".to_owned(),
        window_width: 1000,
        window_height: 1000,
        fullscreen: false,
        window_resizable: true,
        ..Default::default()
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the JSON graph file.
    file: PathBuf,
    /// Wheel sensitivity.
    #[arg(short, long, default_value_t = 0.3)]
    wheel: f64,
    /// Load only nodes with depth <= this (0 = seed). Default 2 keeps the graph interactive.
    #[arg(short, long, default_value_t = 2)]
    depth: u32,
    /// Start with the physics running.
    #[arg(long, default_value_t = false)]
    update: bool,
    /// Initial zoom. 0 (default) fits the initial layout to the window.
    #[arg(short, long, default_value_t = 0.0)]
    zoom: f64,
}

struct Body {
    position: Vec2,
    prev_position: Vec2,
}

impl Body {
    /// A point uniformly on a disc of the given radius (the initial layout: spread, not a blob).
    fn on_disc(radius: f32) -> Self {
        let mut rng = rand::rng();
        let r = radius * rng.random_range(0.0f32..1.0).sqrt();
        let theta = rng.random_range(0.0f32..std::f32::consts::TAU);
        let position = Vec2::new(r * theta.cos(), r * theta.sin());
        Body {
            position,
            prev_position: position,
        }
    }
}

/// The radius of the initial layout disc for `n` nodes.
fn init_radius(n: usize) -> f32 {
    (n as f32).sqrt() + 1.0
}

/// Build a `StableGraph` from a loaded JSON graph. Returns the graph and how many edges were
/// dropped because an endpoint was missing (or the edge was a self-loop).
fn build_graph(data: &GraphData, radius: f32) -> (StableGraph<Body, (), Directed>, usize) {
    let mut graph = StableGraph::<Body, (), Directed>::new();
    let mut id_to_index: HashMap<i64, NodeIndex> = HashMap::with_capacity(data.nodes.len());

    for node in &data.nodes {
        id_to_index.insert(node.id, graph.add_node(Body::on_disc(radius)));
    }

    let mut dropped = 0usize;
    for edge in &data.edges {
        match (id_to_index.get(&edge.source), id_to_index.get(&edge.target)) {
            (Some(&s), Some(&t)) if s != t => {
                graph.add_edge(s, t, ());
            }
            _ => dropped += 1,
        }
    }

    (graph, dropped)
}

/// Zoom that fits a world-space disc of `radius` into the window.
fn fit_zoom(radius: f32, screen_w: f32, screen_h: f32) -> f64 {
    let screen_radius = (screen_w.max(screen_h) / 2.0) as f64;
    0.9 * screen_radius / (radius as f64 * screen_radius)
}

#[macroquad::main(window_conf)]
async fn main() {
    let args = Args::parse();

    let data = match GraphData::from_path(&args.file) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("failed to load {}: {e}", args.file.display());
            std::process::exit(1);
        }
    };
    let data = if args.depth == u32::MAX {
        data
    } else {
        data.filter_depth(args.depth)
    };
    let (node_count, edge_count) = data.counts();
    let radius = init_radius(node_count);

    let (mut graph, dropped) = build_graph(&data, radius);

    let fixed_dt = 1.0 / 120.0;
    let updater = QuadTreeUpdater {
        ..QuadTreeUpdater::default_setting()
    };
    let mut update_running = args.update;

    let mut adaptor = scrollrs::ScrollAdaptor::new(0.0, 0.0, None, None, None);
    adaptor = adaptor.set_zoom(if args.zoom > 0.0 {
        args.zoom
    } else {
        fit_zoom(radius, screen_width(), screen_height())
    });

    // Each node's on-screen size is the examples' `1 + 0.1 * degree`, scaled down so that a hub
    // with thousands of incoming edges stays a circle rather than the whole window.
    let node_unit = (radius as f64) * 1e-3 * 10.0;

    let mut last_mouse_pos = mouse_position();
    let wheel_sensitivity = args.wheel;

    loop {
        let dt = get_frame_time();

        let screen_width = screen_width();
        let screen_height = screen_height();

        adaptor = adaptor.resized(screen_width as f64, screen_height as f64, None, None);

        if is_mouse_button_down(MouseButton::Left) {
            let current_mouse_pos = mouse_position();

            let xrel = current_mouse_pos.0 - last_mouse_pos.0;
            let yrel = current_mouse_pos.1 - last_mouse_pos.1;

            let camera_delta = adaptor.unproject_scalev(DVec2::new(xrel as f64, yrel as f64));
            adaptor = adaptor.moved(-camera_delta);
        }

        let (_wheel_x, wheel_y) = mouse_wheel();
        let zoom_amount = wheel_y as f64 * wheel_sensitivity;
        adaptor = adaptor.zoom(zoom_amount);

        adaptor = adaptor.updated(dt as f64);

        let physical_graph = PhysicalGraph::new(
            graph
                .node_indices()
                .map(|node_index| {
                    let node = PhysicalNode::new(
                        node_index,
                        graph
                            .edges_directed(node_index, Direction::Incoming)
                            .map(|edge_ref| edge_ref.source())
                            .collect::<HashSet<NodeIndex>>(),
                        graph
                            .edges_directed(node_index, Direction::Outgoing)
                            .map(|edge_ref| edge_ref.target())
                            .collect::<HashSet<NodeIndex>>(),
                        graph[node_index].position,
                        graph[node_index].prev_position,
                    );

                    (node_index, node)
                })
                .collect::<HashMap<NodeIndex, PhysicalNode<NodeIndex>>>(),
        );

        if is_key_pressed(KeyCode::Space) {
            update_running = !update_running;
        }

        if update_running {
            let updated_graph = updater.update(&physical_graph, fixed_dt);

            for (node_index, node) in updated_graph.nodes {
                graph[node_index].position = node.position;
                graph[node_index].prev_position = node.prev_position;
            }
        }

        // `f`: fit the current layout to the window.
        if is_key_pressed(KeyCode::F) {
            let mut min = Vec2::splat(f32::MAX);
            let mut max = Vec2::splat(f32::MIN);
            for i in graph.node_indices() {
                min = min.min(graph[i].position);
                max = max.max(graph[i].position);
            }
            let extent = ((max - min).max_element() / 2.0).max(1e-3);
            adaptor = adaptor.set_zoom(fit_zoom(extent, screen_width, screen_height));
        }

        clear_background(color::BLACK);

        let line_width = if 0.5 < adaptor.project_scale(0.002) {
            adaptor.project_scale(0.002)
        } else {
            0.5
        };

        graph.edge_references().for_each(|edge| {
            let source_position = graph[edge.source()].position;
            let target_position = graph[edge.target()].position;

            let source_position = adaptor.project(DVec2::new(
                source_position.x as f64,
                source_position.y as f64,
            ));
            let target_position = adaptor.project(DVec2::new(
                target_position.x as f64,
                target_position.y as f64,
            ));

            draw_line(
                source_position.x as f32,
                source_position.y as f32,
                target_position.x as f32,
                target_position.y as f32,
                line_width as f32,
                color::DARKGRAY,
            );
        });

        graph.node_indices().for_each(|node_index| {
            let position = adaptor.project(DVec2::new(
                graph[node_index].position.x as f64,
                graph[node_index].position.y as f64,
            ));
            let degrees = graph
                .edges_directed(node_index, Direction::Incoming)
                .count();
            let diameter = (1.0 + degrees as f64 * 0.1) * node_unit;

            let projcted_radius = if 0.5 < adaptor.project_scale(diameter * 0.5) {
                adaptor.project_scale(diameter * 0.5)
            } else {
                0.5
            };

            draw_circle(
                position.x as f32,
                position.y as f32,
                projcted_radius as f32,
                color::WHITE,
            )
        });

        draw_text(
            &format!(
                "{}  —  {} nodes, {} edges{}",
                args.file
                    .file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default(),
                node_count,
                edge_count,
                if dropped > 0 {
                    format!(" ({dropped} dangling/self skipped)")
                } else {
                    String::new()
                }
            ),
            20.0,
            20.0,
            20.0,
            color::DARKGRAY,
        );
        draw_text(
            &format!("depth <= {}", args.depth),
            20.0,
            40.0,
            20.0,
            color::DARKGRAY,
        );
        draw_text(
            &format!("FPS: {}", get_fps()),
            20.0,
            60.0,
            20.0,
            color::DARKGRAY,
        );
        draw_text(
            &format!("frame DT: {:.4}", dt),
            20.0,
            80.0,
            20.0,
            color::DARKGRAY,
        );
        draw_text(
            &format!(
                "physics: {}  [space]   pan: drag   zoom: wheel   fit: f",
                if update_running { "ON" } else { "off" }
            ),
            20.0,
            100.0,
            20.0,
            color::DARKGRAY,
        );

        last_mouse_pos = mouse_position();
        next_frame().await
    }
}
