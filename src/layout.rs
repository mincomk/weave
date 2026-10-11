//! A tidy seed layout for a JSON graph: one ring per hop.
//!
//! The updaters refine a graph; they do not invent a shape for it, and from a random cloud they
//! will not find one before you give up looking. For a heavy-tailed graph (a follow graph is the
//! canonical case) that matters: seeded at random, the hubs dominate every force and the result is
//! a hairball. Seeding on concentric rings — ordered around each ring by the mean angle of the
//! node's neighbours on the ring inside it — puts every edge roughly in its radial place, so the
//! picture is readable before any physics runs, and the physics then only has to smooth it.

use crate::json_loader::GraphData;
use glam::Vec2;
use std::collections::{HashMap, VecDeque};

/// Undirected adjacency, as indices into `data.nodes`.
pub fn adjacency_of(data: &GraphData) -> Vec<Vec<u32>> {
    let mut adjacency = vec![Vec::new(); data.nodes.len()];
    let index: HashMap<i64, u32> = data
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.id, i as u32))
        .collect();

    for edge in &data.edges {
        let (Some(&s), Some(&t)) = (index.get(&edge.source), index.get(&edge.target)) else {
            continue;
        };
        if s == t {
            continue;
        }
        adjacency[s as usize].push(t);
        adjacency[t as usize].push(s);
    }

    adjacency
}

/// BFS depth over the undirected view of the graph, from `start`.
pub fn bfs_depths(adjacency: &[Vec<u32>], start: usize) -> Vec<u32> {
    let mut depths = vec![u32::MAX; adjacency.len()];
    let mut queue = VecDeque::new();
    depths[start] = 0;
    queue.push_back(start);

    while let Some(u) = queue.pop_front() {
        for &v in &adjacency[u] {
            let v = v as usize;
            if depths[v] == u32::MAX {
                depths[v] = depths[u] + 1;
                queue.push_back(v);
            }
        }
    }

    depths
}

/// Hop depth per node: the file's own `depth` when every node carries one, BFS otherwise.
pub fn depths_of(data: &GraphData, adjacency: &[Vec<u32>]) -> Vec<u32> {
    let complete = !data.nodes.is_empty() && data.nodes.iter().all(|n| n.depth.is_some());
    if complete {
        return data.nodes.iter().map(|n| n.depth.unwrap()).collect();
    }

    let mut depths = vec![u32::MAX; data.nodes.len()];
    if data.nodes.is_empty() {
        return depths;
    }

    while let Some(start) = depths.iter().position(|&d| d == u32::MAX) {
        let component = bfs_depths(adjacency, start);
        for (i, d) in component.iter().enumerate() {
            if depths[i] == u32::MAX {
                depths[i] = *d;
            }
        }
    }

    depths
}

/// One ring per hop, evenly spaced around it, ordered by the mean angle of the node's parents so
/// that edges run outward instead of across.
pub fn tidy_layout(depths: &[u32], adjacency: &[Vec<u32>]) -> Vec<Vec2> {
    let n = depths.len();
    let mut positions = vec![Vec2::ZERO; n];
    let mut angles = vec![0.0f32; n];
    if n == 0 {
        return positions;
    }

    let max_depth = depths.iter().copied().max().unwrap_or(0);
    let mut rings: Vec<Vec<usize>> = vec![Vec::new(); max_depth as usize + 1];
    for (i, &d) in depths.iter().enumerate() {
        rings[d as usize].push(i);
    }

    const ARC: f32 = 1.0;
    let tau = std::f32::consts::TAU;

    for d in 1..=max_depth {
        let ring = &rings[d as usize];
        let count = ring.len();
        if count == 0 {
            continue;
        }

        // Exactly enough circumference for the nodes on the ring, never tighter than one step.
        let radius = (ARC * count as f32 / tau).max(ARC * d as f32);

        let mut ordered: Vec<(f32, usize)> = ring
            .iter()
            .map(|&i| {
                let (mut sx, mut sy) = (0.0f32, 0.0f32);
                for &j in &adjacency[i] {
                    if depths[j as usize] == d - 1 {
                        let a = angles[j as usize];
                        sx += a.cos();
                        sy += a.sin();
                    }
                }
                let proposed = if sx == 0.0 && sy == 0.0 {
                    tau * (i as f32) / (count as f32) // no parent on the previous ring
                } else {
                    sy.atan2(sx)
                };
                (proposed, i)
            })
            .collect();

        ordered.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        for (slot, &(_, i)) in ordered.iter().enumerate() {
            let angle = tau * slot as f32 / count as f32;
            positions[i] = Vec2::new(radius * angle.cos(), radius * angle.sin());
            angles[i] = angle;
        }
    }

    positions
}

/// Convenience: adjacency, depths and a tidy layout in one call.
pub fn seed_layout(data: &GraphData) -> (Vec<Vec2>, Vec<u32>) {
    let adjacency = adjacency_of(data);
    let depths = depths_of(data, &adjacency);
    let positions = tidy_layout(&depths, &adjacency);
    (positions, depths)
}
