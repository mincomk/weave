//! Interactive visualiser for a JSON graph, driven by `QuadTreeUpdater` and `scrollrs`.
//!
//! * left-drag  — pan
//! * wheel      — zoom
//! * space      — start / stop the physics
//! * `f`        — fit the whole graph to the window
//! * `r`        — reset to the tidy (ring) layout
//! * `e`        — toggle edges
//! * `c`        — clear the current highlight
//! * `/`        — open the search box; type a login, `enter` to select, `esc` to clear
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
//! * **Labels that read.** The busiest nodes are labelled with a dark backing plate behind the text,
//!   which is what keeps a login legible where a dozen edges cross it.
//! * **Search and highlight.** Type a login into the search box and that account plus its one-hop
//!   neighbourhood are picked out and coloured by the *relation* to it — accounts it follows, the
//!   accounts that follow it, and the accounts that do both — and the camera frames them.

use clap::Parser;
use glam::{DVec2, Vec2};
use macroquad::{
    color::{self, Color},
    input::{
        KeyCode, MouseButton, get_char_pressed, is_key_pressed, is_mouse_button_down,
        mouse_position, mouse_wheel,
    },
    shapes::{draw_circle, draw_line, draw_rectangle, draw_rectangle_lines},
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
/// Node-label font size. Larger than the HUD: a login is read against a moving graph, not a page.
const LABEL_SIZE: f32 = 19.0;
const MIN_EDGE_PX: f64 = 1.5;
const MIN_LABEL_RADIUS_PX: f32 = 4.0;
const MAX_NODE_RADIUS_PX: f32 = 14.0;
/// Longest login a search box accepts.
const SEARCH_MAX_CHARS: usize = 39;
/// A hub can have thousands of neighbours; past this many the labels cannot fit anyway, so only the
/// busiest are labelled.
const MAX_NEIGHBOUR_LABELS: usize = 200;

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
    #[arg(long, default_value_t = 60)]
    labels: usize,
    /// Login to highlight at startup (as if it had been searched).
    #[arg(long)]
    search: Option<String>,
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

/// How a highlighted account relates to the searched one, read off the *directed* edges.
///
/// An edge `source -> target` reads as "source follows target" (that is the direction the loader
/// documents and the one the physics relies on, since in-degree is mass).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Relation {
    /// The searched account follows them (outgoing edge only).
    Following,
    /// They follow the searched account (incoming edge only).
    Follower,
    /// Both directions.
    Mutual,
}

fn relation_color(r: Relation) -> Color {
    match r {
        Relation::Following => Color::new(1.00, 0.63, 0.15, 1.0), // amber
        Relation::Follower => Color::new(0.36, 0.64, 1.00, 1.0),  // blue
        Relation::Mutual => Color::new(0.24, 0.86, 0.46, 1.0),    // green
    }
}

fn relation_label(r: Relation) -> &'static str {
    match r {
        Relation::Following => "following (seed → them)",
        Relation::Follower => "follower (them → seed)",
        Relation::Mutual => "mutual (both ways)",
    }
}

fn selected_color() -> Color {
    Color::new(1.00, 0.98, 0.55, 1.0)
}

/// The searched account and its classified one-hop neighbourhood.
struct Selection {
    node: NodeIndex,
    label: String,
    rel: HashMap<NodeIndex, Relation>,
    following: usize,
    follower: usize,
    mutual: usize,
}

impl Selection {
    fn of(graph: &StableGraph<Body, (), Directed>, node: NodeIndex, label: String) -> Self {
        let outgoing: HashSet<NodeIndex> = graph
            .edges_directed(node, Direction::Outgoing)
            .map(|e| e.target())
            .filter(|t| *t != node)
            .collect();
        let incoming: HashSet<NodeIndex> = graph
            .edges_directed(node, Direction::Incoming)
            .map(|e| e.source())
            .filter(|s| *s != node)
            .collect();

        let mut rel: HashMap<NodeIndex, Relation> = HashMap::new();
        for &v in &outgoing {
            rel.insert(v, Relation::Following);
        }
        let mut mutual = 0usize;
        let mut follower = 0usize;
        for &v in &incoming {
            match rel.get_mut(&v) {
                // It was already Following, so both directions exist.
                Some(slot) => {
                    *slot = Relation::Mutual;
                    mutual += 1;
                }
                None => {
                    rel.insert(v, Relation::Follower);
                    follower += 1;
                }
            }
        }

        Selection {
            node,
            label,
            rel,
            following: outgoing.len() - mutual,
            follower,
            mutual,
        }
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

/// Resolve a typed login to a node: an exact (case-insensitive) match wins, otherwise the busiest
/// account whose login starts with what was typed.
fn find_login(
    query: &str,
    login_to_index: &HashMap<String, NodeIndex>,
    degree_of: &HashMap<NodeIndex, usize>,
) -> Option<NodeIndex> {
    let q = query.trim().to_lowercase();
    if q.is_empty() {
        return None;
    }
    if let Some(&i) = login_to_index.get(&q) {
        return Some(i);
    }
    login_to_index
        .iter()
        .filter(|(login, _)| login.starts_with(&q))
        .max_by_key(|(_, i)| {
            (
                degree_of.get(i).copied().unwrap_or(0),
                std::cmp::Reverse(i.index()),
            )
        })
        .map(|(_, &i)| i)
}

/// Draw one label with a dark backing plate, unless it would land on one already placed. Returns
/// whether it was drawn.
fn draw_label(
    text: &str,
    position: Vec2,
    radius: f32,
    color: Color,
    size: f32,
    plate: Color,
    placed: &mut Vec<(f32, f32, f32, f32)>,
) -> bool {
    let measured = measure_text(text, None, size as u16, 1.0);
    let x = position.x + radius + 5.0;
    let y = position.y + measured.offset_y * 0.5;
    let pad = 3.0;
    let box_ = (
        x - pad,
        y - measured.height - pad,
        measured.width + pad * 2.0,
        measured.height + pad * 2.0,
    );
    if placed.iter().any(|&(px, py, pw, ph)| {
        box_.0 < px + pw && px < box_.0 + box_.2 && box_.1 < py + ph && py < box_.1 + box_.3
    }) {
        return false;
    }
    placed.push(box_);

    // The plate is what makes the text legible over the edges and nodes beneath it — a thin outline
    // disappears into a dense core, a filled rectangle does not.
    draw_rectangle(box_.0, box_.1, box_.2, box_.3, plate);
    draw_text(text, x, y, size, color);
    true
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
    let degree_of: HashMap<NodeIndex, usize> = degrees.iter().map(|(d, i)| (*i, *d)).collect();
    let label_of: Vec<Option<&str>> = data.nodes.iter().map(|n| n.label.as_deref()).collect();
    // Login -> node, for the search box. Node `i` of the loaded data is node `i` of the graph
    // (`build_graph` adds them in order), so an index here is directly a `NodeIndex`.
    let login_to_index: HashMap<String, NodeIndex> = data
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(i, n)| {
            n.label
                .as_ref()
                .map(|l| (l.to_lowercase(), NodeIndex::new(i)))
        })
        .collect();
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

    // --- search state -------------------------------------------------------
    let mut query = String::new();
    let mut search_focused = false;
    let mut search_error = false;
    let mut selection: Option<Selection> = None;
    // While this is set the camera keeps the highlighted group framed; any pan or wheel hands
    // the camera back to the user.
    let mut follow_selection = false;
    if let Some(login) = &args.search
        && let Some(node) = find_login(login, &login_to_index, &degree_of)
    {
        selection = Some(Selection::of(
            &graph,
            node,
            label_of[node.index()].unwrap_or("").to_string(),
        ));
        follow_selection = true;
        auto_fit = false; // hand the camera to the selection rather than the whole graph
    }

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
                follow_selection = false;
            }
            adaptor = adaptor.moved(-camera_delta);
        }

        let (_wheel_x, wheel_y) = mouse_wheel();
        if wheel_y != 0.0 {
            auto_fit = false;
            follow_selection = false;
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

        // --- search box input ---------------------------------------------------
        // `/` opens it (the slash that opened it is never the first character of a login), Enter
        // resolves, Escape clears. While it is focused the single-key shortcuts are suppressed so
        // typing a login cannot toggle the edges or restart the physics under you.
        if is_key_pressed(KeyCode::Slash) && !search_focused {
            search_focused = true;
            search_error = false;
        }
        // Drained every frame, not only while the box has focus: macroquad's character queue is not
        // cleared per frame, so a key pressed while the box was closed would otherwise land in the
        // query the moment it opened.
        let mut typed: Vec<char> = Vec::new();
        while let Some(c) = get_char_pressed() {
            typed.push(c);
        }

        if search_focused {
            for c in typed {
                if c == '/' {
                    continue; // a slash opens the box; it is never part of a login
                }
                if !c.is_control() && query.chars().count() < SEARCH_MAX_CHARS {
                    query.push(c);
                    search_error = false;
                }
            }
            if is_key_pressed(KeyCode::Backspace) {
                query.pop();
                search_error = false;
            }
            if is_key_pressed(KeyCode::Escape) {
                search_focused = false;
                query.clear();
                search_error = false;
            }
            if is_key_pressed(KeyCode::Enter) {
                match find_login(&query, &login_to_index, &degree_of) {
                    Some(node) => {
                        selection = Some(Selection::of(
                            &graph,
                            node,
                            label_of[node.index()].unwrap_or("").to_string(),
                        ));
                        follow_selection = true;
                        auto_fit = false; // the camera follows the group from here
                        search_focused = false;
                        search_error = false;
                    }
                    // Nothing matched: keep whatever was already highlighted and say so, rather
                    // than throwing away a good answer because of a typo.
                    None => search_error = true,
                }
            }
        }

        // --- shortcuts (only when the search box is not taking the keyboard) -----
        if !search_focused {
            if is_key_pressed(KeyCode::Space) {
                update_running = !update_running;
            }

            if is_key_pressed(KeyCode::E) {
                hide_edges = !hide_edges;
            }

            if is_key_pressed(KeyCode::F) {
                auto_fit = true;
                follow_selection = false;
            }

            if is_key_pressed(KeyCode::R) {
                let indices: Vec<NodeIndex> = graph.node_indices().collect();
                for (i, node_index) in indices.into_iter().enumerate() {
                    graph[node_index].position = positions[i];
                    graph[node_index].prev_position = positions[i];
                }
                update_running = false;
            }

            if is_key_pressed(KeyCode::C) {
                selection = None;
                query.clear();
                follow_selection = false;
            }
        }

        // Keep the selected account and its neighbourhood framed while the physics is still
        // moving. It has to keep being re-framed rather than framed once: a search that lands on
        // the seed layout would otherwise be aimed at where the nodes were before the layout
        // spread them.
        if follow_selection && !auto_fit {
            match &selection {
                Some(sel) => {
                    let mut points: Vec<Vec2> = Vec::with_capacity(sel.rel.len() + 1);
                    points.push(graph[sel.node].position);
                    for v in sel.rel.keys() {
                        points.push(graph[*v].position);
                    }
                    let center =
                        points.iter().fold(Vec2::ZERO, |acc, p| acc + *p) / points.len() as f32;
                    let radius = points
                        .iter()
                        .map(|p| (*p - center).length())
                        .fold(1.0f32, f32::max);

                    let camera = adaptor.unproject(DVec2::new(
                        screen_width as f64 / 2.0,
                        screen_height as f64 / 2.0,
                    ));
                    adaptor = adaptor.moved(DVec2::new(
                        center.x as f64 - camera.x,
                        center.y as f64 - camera.y,
                    ));
                    adaptor = adaptor.set_zoom(fit_zoom(radius * 1.3));
                    auto_fit_center = center;
                }
                None => follow_selection = false,
            }
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
            let edge_color = Color::new(
                0.62,
                0.66,
                0.74,
                if selection.is_some() { 0.07 } else { 0.30 },
            );
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
                    a.x as f32, a.y as f32, b.x as f32, b.y as f32, 1.0, edge_color,
                );
            });

            // The edges between the selected account and its neighbourhood, coloured by relation.
            if let Some(sel) = &selection {
                let a = project(graph[sel.node].position);
                for (v, rel) in &sel.rel {
                    let b = project(graph[*v].position);
                    let mut c = relation_color(*rel);
                    c.a = 0.55;
                    draw_line(a.x as f32, a.y as f32, b.x as f32, b.y as f32, 1.6, c);
                }
            }
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
        let pos_of: HashMap<NodeIndex, (Vec2, f32)> =
            radii.iter().map(|(i, p, r, _)| (*i, (*p, *r))).collect();

        // Everything not in the selection is dimmed so the highlighted set reads immediately.
        for (node_index, position, radius, depth) in &radii {
            if let Some(sel) = &selection {
                if *node_index == sel.node || sel.rel.contains_key(node_index) {
                    continue; // drawn below, on top
                }
                let mut c = depth_color(*depth);
                c.a = 0.14;
                draw_circle(position.x, position.y, *radius, c);
            } else {
                draw_circle(position.x, position.y, *radius, depth_color(*depth));
            }
        }

        if let Some(sel) = &selection {
            for (v, rel) in &sel.rel {
                if let Some((position, radius)) = pos_of.get(v) {
                    draw_circle(
                        position.x,
                        position.y,
                        radius + 2.0,
                        Color::new(0.02, 0.03, 0.05, 0.9),
                    );
                    draw_circle(position.x, position.y, radius + 1.0, relation_color(*rel));
                }
            }
            if let Some((position, radius)) = pos_of.get(&sel.node) {
                draw_circle(
                    position.x,
                    position.y,
                    radius + 4.5,
                    Color::new(1.0, 1.0, 1.0, 0.95),
                );
                draw_circle(position.x, position.y, radius + 2.5, selected_color());
                draw_circle(
                    position.x,
                    position.y,
                    *radius,
                    Color::new(0.08, 0.09, 0.12, 1.0),
                );
            }
        }

        // --- labels ------------------------------------------------------------
        // The selection and its neighbours are labelled first (and regardless of `--labels`): the
        // point of the search is to read those logins. Then the busiest nodes fill in what is left,
        // a label that would land on one already placed being dropped.
        let mut placed: Vec<(f32, f32, f32, f32)> = Vec::new();
        let plate = Color::new(0.02, 0.03, 0.05, 0.78);

        if let Some(sel) = &selection {
            if let Some((position, radius)) = pos_of.get(&sel.node) {
                draw_label(
                    &sel.label,
                    *position,
                    *radius,
                    selected_color(),
                    LABEL_SIZE + 2.0,
                    plate,
                    &mut placed,
                );
            }

            let mut neighbours: Vec<(NodeIndex, Relation)> =
                sel.rel.iter().map(|(k, v)| (*k, *v)).collect();
            neighbours
                .sort_by_key(|(i, _)| std::cmp::Reverse(degree_of.get(i).copied().unwrap_or(0)));
            for (v, rel) in neighbours.into_iter().take(MAX_NEIGHBOUR_LABELS) {
                let Some((position, radius)) = pos_of.get(&v) else {
                    continue;
                };
                let text = label_of.get(v.index()).copied().flatten().unwrap_or("");
                if text.is_empty() {
                    continue;
                }
                draw_label(
                    text,
                    *position,
                    *radius,
                    relation_color(rel),
                    LABEL_SIZE,
                    plate,
                    &mut placed,
                );
            }
        }

        for (node_index, label) in &labels {
            if selection
                .as_ref()
                .map(|s| s.node == *node_index || s.rel.contains_key(node_index))
                .unwrap_or(false)
            {
                continue; // already drawn above
            }
            let Some((position, radius)) = pos_of.get(node_index) else {
                continue;
            };
            if *radius < MIN_LABEL_RADIUS_PX {
                continue;
            }
            draw_label(
                label,
                *position,
                *radius,
                Color::new(0.95, 0.95, 0.95, 1.0),
                LABEL_SIZE,
                plate,
                &mut placed,
            );
        }

        // --- hud ---------------------------------------------------------------
        let mut lines: Vec<String> = vec![
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
                if auto_fit {
                    "auto  "
                } else if follow_selection && selection.is_some() {
                    "follow"
                } else {
                    "manual"
                },
                if update_running { physics_ms } else { 0.0 }
            ),
            format!(
                "{} fps   {:.4} ms/frame   [space] run  [f] fit  [r] reset  [e] edges  [/] search  [c] clear",
                get_fps(),
                dt * 1e3
            ),
            format!(
                "drag pan · wheel zoom · {} edges",
                if hide_edges { "no" } else { "with" }
            ),
        ];
        if let Some(sel) = &selection {
            lines.push(format!(
                "@{} — following {} · followers {} · mutual {}",
                sel.label, sel.following, sel.follower, sel.mutual
            ));
        }

        let width = lines
            .iter()
            .map(|l| measure_text(l, None, FONT as u16, 1.0).width)
            .fold(0.0f32, f32::max);
        draw_rectangle(
            0.0,
            0.0,
            width + 28.0,
            lines.len() as f32 * (FONT + 6.0) + 16.0,
            Color::new(0.0, 0.0, 0.0, 0.72),
        );
        for (i, line) in lines.iter().enumerate() {
            let color = if i + 1 == lines.len() && selection.is_some() {
                selected_color()
            } else {
                Color::new(0.82, 0.85, 0.90, 1.0)
            };
            draw_text(
                line,
                14.0,
                16.0 + (i as f32 + 0.8) * (FONT + 6.0),
                FONT,
                color,
            );
        }

        // --- search box ---------------------------------------------------------
        let box_w = 400.0f32;
        let box_h = 34.0f32;
        let bx = (screen_width - box_w) / 2.0;
        let by = 12.0;
        draw_rectangle(bx, by, box_w, box_h, Color::new(0.04, 0.05, 0.08, 0.94));
        draw_rectangle_lines(
            bx,
            by,
            box_w,
            box_h,
            2.0,
            if search_focused {
                Color::new(0.45, 0.70, 1.00, 1.0)
            } else {
                Color::new(0.26, 0.31, 0.41, 1.0)
            },
        );
        let (shown, shown_color) = if query.is_empty() && !search_focused {
            (
                "search:  press  /  and type a login".to_string(),
                Color::new(0.50, 0.55, 0.66, 1.0),
            )
        } else {
            (
                format!(
                    "search:  {}{}",
                    query,
                    if search_focused { "_" } else { "" }
                ),
                Color::new(0.96, 0.97, 1.00, 1.0),
            )
        };
        draw_text(&shown, bx + 10.0, by + 23.0, 18.0, shown_color);

        // The failure message goes to the right of the box, clear of the HUD panel in the corner.
        if search_error {
            draw_text(
                &format!("no account matches \"{query}\""),
                bx + box_w + 12.0,
                by + 23.0,
                16.0,
                relation_color(Relation::Following),
            );
        }

        // --- legend -------------------------------------------------------------
        if selection.is_some() {
            let legend = [Relation::Following, Relation::Follower, Relation::Mutual];
            let y0 = screen_height - 12.0 - legend.len() as f32 * 22.0;
            for (i, rel) in legend.iter().enumerate() {
                let y = y0 + i as f32 * 22.0;
                draw_circle(18.0, y, 6.0, relation_color(*rel));
                draw_text(
                    relation_label(*rel),
                    30.0,
                    y + 5.0,
                    16.0,
                    Color::new(0.90, 0.92, 0.96, 1.0),
                );
            }
        }

        last_mouse_pos = mouse_position();
        next_frame().await
    }
}
