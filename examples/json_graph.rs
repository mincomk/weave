//! Interactive visualiser for a JSON graph, driven by `QuadTreeUpdater` and `scrollrs`.
//!
//! * left-drag  — pan
//! * wheel      — zoom
//! * space      — start / stop the physics
//! * `f`        — fit the whole graph to the window
//! * `r`        — reset to the tidy (ring) layout
//!
//! ```sh
//! cargo run -r --example json_graph -- follow-graph-weave.json -w 0.3 --depth 2
//! cargo run -r --example json_graph -- follow-graph-weave.json -w 0.3 --depth 3 --hide-edges
//! ```
//!
//! The scroll/camera handling is the same as `node_physics` / `file_tree`. What is different is
//! everything on top of it, because a real graph with a long tail of hubs is unreadable drawn the
//! way the small examples draw:
//!
//! * **Tidy layout** ([`weave::layout`]). Nodes are seeded on rings by hop depth, ordered around
//!   each ring by the mean angle of their neighbours, so the first frame is structure rather than
//!   confetti and the physics starts from something rather than from a random cloud.
//! * **The physics runs, and the view follows it.** The layout spread is settled by
//!   `QuadTreeUpdater::unit_scale_setting` while the camera auto-fits the graph to the window;
//!   panning or zooming hands the camera to you, and `f` hands it back.
//! * **Edges you can actually see.** An edge shorter than ~1.5 px on screen is not drawn (at that
//!   scale it is a smudge, and forty thousand of them are the grey blob), nor is one whose endpoints
//!   are both off screen. The rest are drawn dim, so they read as structure and not as ink.
//! * **Node size clamped in screen space**, so a hub with ten thousand followers is a small disc
//!   rather than something larger than the window, and hue carries the hop depth.
//! * **The busiest nodes are labelled** (up to `--labels`, drawn once they are big enough on screen).

use clap::Parser;
use glam::{DVec2, Vec2};
use macroquad::{
    color::{self, Color},
    input::{
        KeyCode, MouseButton, is_key_pressed, is_mouse_button_down, mouse_position, mouse_wheel,
    },
    shapes::{draw_circle, draw_line, draw_rectangle},
    text::{draw_text, measure_text},
    time::{get_fps, get_frame_time},
    window::{Conf, clear_background, next_frame, screen_height, screen_width},
};
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use petgraph::{
    Directed, Direction,
    stable_graph::{NodeIndex, StableGraph},
};
use scrollrs::Projector;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use weave::json_loader::GraphData;
use weave::layout::seed_layout;
use weave::updater::{QuadTreeUpdater, Updater};
use weave::{PhysicalGraph, PhysicalNode};

const FONT: f32 = 18.0;
const MIN_EDGE_PX: f64 = 1.5;
const MIN_LABEL_RADIUS_PX: f32 = 4.0;
const MAX_NODE_RADIUS_PX: f32 = 14.0;

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
    /// Start with the physics frozen (it runs by default — it is what spreads the graph out).
    #[arg(long, default_value_t = false)]
    no_update: bool,
    /// Do not start the physics above this many nodes (a tick is O(n^2)).
    #[arg(long, default_value_t = 5000)]
    max_auto_nodes: usize,
    /// Initial zoom. 0 (default) fits the layout to the window.
    #[arg(short, long, default_value_t = 0.0)]
    zoom: f64,
    /// Do not draw edges (useful on the full graph).
    #[arg(long, default_value_t = false)]
    hide_edges: bool,
    /// Hide the edges at startup above this many nodes (press `e` to toggle them back).
    ///
    /// Past a few tens of thousands of long edges they stop being structure and become a filled
    /// disc: every one of them is drawn and each is hundreds of pixels long.
    #[arg(long, default_value_t = 20000)]
    max_edge_nodes: usize,
    /// Label at most this many of the busiest nodes (0 = none).
    #[arg(long, default_value_t = 40)]
    labels: usize,
}

struct Body {
    position: Vec2,
    prev_position: Vec2,
}

/// Colour by hop depth: seed, then the rings outward.
fn depth_color(depth: u32) -> Color {
    match depth {
        0 => Color::new(1.00, 0.83, 0.25, 1.0),
        1 => Color::new(0.40, 0.72, 1.00, 1.0),
        2 => Color::new(0.50, 0.88, 0.58, 1.0),
        3 => Color::new(0.80, 0.62, 0.95, 1.0),
        _ => Color::new(0.78, 0.78, 0.82, 1.0),
    }
}

/// Build a `StableGraph` from a loaded JSON graph. Returns the graph and how many edges were
/// dropped because an endpoint was missing (or the edge was a self-loop).
fn build_graph(data: &GraphData, positions: &[Vec2]) -> (StableGraph<Body, (), Directed>, usize) {
    let mut graph = StableGraph::<Body, (), Directed>::new();
    let mut id_to_index: HashMap<i64, NodeIndex> = HashMap::with_capacity(data.nodes.len());

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
fn fit_zoom(radius: f32) -> f64 {
    0.9 / (radius.max(1e-3) as f64)
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

    let (positions, depths) = seed_layout(&data);
    let extent = positions
        .iter()
        .map(|p| p.length())
        .fold(0.0f32, f32::max)
        .max(1.0);

    let (mut graph, dropped) = build_graph(&data, &positions);

    // Node size follows the examples' `1 + 0.1 * degree`, but as a fraction of the layout so it
    // keeps its on-screen proportion whatever the graph's extent, and clamped below.
    let node_unit = extent * 2e-3;

    let fixed_dt = 1.0 / 120.0;
    // The unit-scale setting: this layout keeps neighbours a few units apart, and the graph has a
    // long tail of hubs, which is exactly what `default_setting()` is not tuned for.
    let updater = QuadTreeUpdater::unit_scale_setting();
    let mut update_running = !args.no_update && node_count <= args.max_auto_nodes;
    if !args.no_update && node_count > args.max_auto_nodes {
        eprintln!(
            "{node_count} nodes is above --max-auto-nodes ({}): starting with the physics frozen. \
             Press space to run it anyway, or pass a smaller --depth.",
            args.max_auto_nodes
        );
    }
    // While the view is auto-fitting it keeps the whole graph in the window as the physics spreads
    // it; touching the camera (or `f` to hand it back) makes the view yours.
    let mut auto_fit = true;
    let mut auto_fit_center = Vec2::ZERO;

    let mut adaptor = scrollrs::ScrollAdaptor::new(0.0, 0.0, None, None, None);
    adaptor = adaptor.set_zoom(if args.zoom > 0.0 {
        args.zoom
    } else {
        fit_zoom(extent)
    });

    // The busiest nodes, most important first, so the label pass can drop the ones that would
    // collide with a more important one.
    let mut degrees: Vec<(usize, NodeIndex)> = graph
        .node_indices()
        .map(|i| (graph.edges_directed(i, Direction::Incoming).count(), i))
        .collect();
    degrees.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.index().cmp(&b.1.index())));
    let label_of: Vec<Option<&str>> = data.nodes.iter().map(|n| n.label.as_deref()).collect();
    let labels: Vec<(NodeIndex, String)> = degrees
        .iter()
        .take(args.labels)
        .filter(|(degree, _)| *degree > 0)
        .filter_map(|(_, index)| {
            label_of
                .get(index.index())
                .copied()
                .flatten()
                .map(|label| (*index, label.to_string()))
        })
        .collect();

    let mut hide_edges = args.hide_edges;
    if !args.hide_edges && node_count > args.max_edge_nodes {
        eprintln!(
            "{node_count} nodes is above --max-edge-nodes ({}): starting with the edges hidden, \
             because that many long edges fill the window with ink. Press e to show them.",
            args.max_edge_nodes
        );
        hide_edges = true;
    }

    let mut last_mouse_pos = mouse_position();
    let wheel_sensitivity = args.wheel;
    let mut physics_ms = -1.0f64;

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
            if xrel != 0.0 || yrel != 0.0 {
                auto_fit = false;
            }
            adaptor = adaptor.moved(-camera_delta);
        }

        let (_wheel_x, wheel_y) = mouse_wheel();
        if wheel_y != 0.0 {
            auto_fit = false;
        }
        let zoom_amount = wheel_y as f64 * wheel_sensitivity;
        adaptor = adaptor.zoom(zoom_amount);

        adaptor = adaptor.updated(dt as f64);

        if auto_fit {
            // Fit the body of the graph, not its strays: a handful of nodes flung far out would
            // otherwise shrink everything else to a speck. Recentre on the centroid and frame the
            // p99 radius, which is a stable frame while the physics is still moving.
            let mut center = Vec2::ZERO;
            let mut count = 0.0f32;
            for node_index in graph.node_indices() {
                center += graph[node_index].position;
                count += 1.0;
            }
            if count > 0.0 {
                center /= count;
                let mut radii: Vec<f32> = graph
                    .node_indices()
                    .map(|i| (graph[i].position - center).length())
                    .collect();
                radii.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let p99 = radii[(radii.len() * 99 / 100).min(radii.len() - 1)];

                adaptor = adaptor.moved(DVec2::new(
                    (center.x - auto_fit_center.x) as f64,
                    (center.y - auto_fit_center.y) as f64,
                ));
                auto_fit_center = center;

                adaptor = adaptor.set_zoom(fit_zoom(p99.max(1e-3) * 1.15));
            }
        }

        if is_key_pressed(KeyCode::Space) {
            update_running = !update_running;
        }

        if is_key_pressed(KeyCode::E) {
            hide_edges = !hide_edges;
        }

        if is_key_pressed(KeyCode::F) {
            auto_fit = true;
        }

        if is_key_pressed(KeyCode::R) {
            let indices: Vec<NodeIndex> = graph.node_indices().collect();
            for (i, node_index) in indices.into_iter().enumerate() {
                graph[node_index].position = positions[i];
                graph[node_index].prev_position = positions[i];
            }
            update_running = false;
        }

        if update_running {
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

            let started = std::time::Instant::now();
            let updated_graph = updater.update(&physical_graph, fixed_dt);
            physics_ms = started.elapsed().as_secs_f64() * 1e3;

            for (node_index, node) in updated_graph.nodes {
                graph[node_index].position = node.position;
                graph[node_index].prev_position = node.prev_position;
            }
        }

        clear_background(color::BLACK);

        let project = |p: Vec2| adaptor.project(DVec2::new(p.x as f64, p.y as f64));
        let on_screen = |p: DVec2| {
            p.x > -64.0
                && p.x < screen_width as f64 + 64.0
                && p.y > -64.0
                && p.y < screen_height as f64 + 64.0
        };

        // --- edges -------------------------------------------------------------
        if !hide_edges {
            let edge_color = Color::new(0.62, 0.66, 0.74, 0.30);
            graph.edge_references().for_each(|edge| {
                let a = project(graph[edge.source()].position);
                let b = project(graph[edge.target()].position);
                if !on_screen(a) && !on_screen(b) {
                    return;
                }
                // At this scale a shorter edge is a smudge, and a canvas of smudges is a blob.
                if (b - a).length_squared() < MIN_EDGE_PX * MIN_EDGE_PX {
                    return;
                }

                draw_line(
                    a.x as f32,
                    a.y as f32,
                    b.x as f32,
                    b.y as f32,
                    1.0,
                    edge_color,
                );
            });
        }

        // --- nodes -------------------------------------------------------------
        let mut radii: Vec<(NodeIndex, Vec2, f32, u32)> = Vec::new();
        for node_index in graph.node_indices() {
            let position = project(graph[node_index].position);
            if !on_screen(position) {
                continue;
            }
            let degree = graph
                .edges_directed(node_index, Direction::Incoming)
                .count();
            let world_diameter = (1.0 + degree as f32 * 0.1) * node_unit;
            let radius = (adaptor.project_scale(world_diameter as f64 * 0.5) as f32)
                .clamp(1.0, MAX_NODE_RADIUS_PX);

            radii.push((
                node_index,
                Vec2::new(position.x as f32, position.y as f32),
                radius,
                depths[node_index.index()],
            ));
        }

        for (_, position, radius, depth) in &radii {
            draw_circle(position.x, position.y, *radius, depth_color(*depth));
        }

        // --- labels ------------------------------------------------------------
        // Busiest first; a label that would land on one already placed is dropped, which is what
        // keeps the dense core readable instead of a pile of overlapping words.
        let mut placed: Vec<(f32, f32, f32, f32)> = Vec::new();
        for (node_index, label) in &labels {
            let Some((_, position, radius, _)) = radii.iter().find(|(i, ..)| i == node_index) else {
                continue;
            };
            if *radius < MIN_LABEL_RADIUS_PX {
                continue;
            }

            let size = 14.0;
            let measured = measure_text(label, None, size as u16, 1.0);
            let x = position.x + radius + 4.0;
            let y = position.y + measured.offset_y * 0.5;
            let box_ = (
                x - 1.0,
                y - measured.height,
                measured.width + 2.0,
                measured.height + 1.0,
            );
            if placed.iter().any(|&(px, py, pw, ph)| {
                box_.0 < px + pw && px < box_.0 + box_.2 && box_.1 < py + ph && py < box_.1 + box_.3
            }) {
                continue;
            }
            placed.push(box_);

            for (dx, dy) in [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)] {
                draw_text(label, x + dx, y + dy, size, Color::new(0.0, 0.0, 0.0, 0.85));
            }
            draw_text(label, x, y, size, Color::new(0.95, 0.95, 0.95, 1.0));
        }

        // --- hud ---------------------------------------------------------------
        let lines = vec![
            format!(
                "{}   {} nodes · {} edges{}",
                args.file
                    .file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default(),
                node_count,
                edge_count,
                if dropped > 0 {
                    format!(" · {dropped} dropped")
                } else {
                    String::new()
                }
            ),
            format!(
                "hop <= {}   physics {}   fit {}   {:.4} ms/tick",
                args.depth,
                if update_running { "on " } else { "off" },
                if auto_fit { "auto  " } else { "manual" },
                if update_running { physics_ms } else { 0.0 }
            ),
            format!(
                "{} fps   {:.4} ms/frame   [space] run  [f] fit  [r] reset",
                get_fps(),
                dt * 1e3
            ),
            format!(
                "drag pan · wheel zoom · {} edges",
                if hide_edges { "no" } else { "with" }
            ),
        ];

        let width = lines
            .iter()
            .map(|l| measure_text(l, None, FONT as u16, 1.0).width)
            .fold(0.0f32, f32::max);
        draw_rectangle(
            0.0,
            0.0,
            width + 28.0,
            lines.len() as f32 * (FONT + 6.0) + 16.0,
            Color::new(0.0, 0.0, 0.0, 0.55),
        );
        for (i, line) in lines.iter().enumerate() {
            draw_text(
                line,
                14.0,
                16.0 + (i as f32 + 0.8) * (FONT + 6.0),
                FONT,
                Color::new(0.72, 0.75, 0.80, 1.0),
            );
        }

        last_mouse_pos = mouse_position();
        next_frame().await
    }
}
