//! A dependency-light JSON graph loader.
//!
//! The rest of `weave` works on an in-memory [`crate::PhysicalGraph`]; this module is the bridge
//! from a JSON file on disk to that. The wire format is deliberately flat and generic, so any
//! node/edge graph can be dumped from any language:
//!
//! ```json
//! {
//!   "directed": true,
//!   "nodes": [
//!     { "id": 0, "label": "ityeri", "depth": 0 },
//!     { "id": 1, "label": "Jung2010", "depth": 1 }
//!   ],
//!   "edges": [
//!     { "source": 1, "target": 0 }
//!   ]
//! }
//! ```
//!
//! `id` is a stable integer identity. `label` and `depth` are optional metadata (carried through
//! for the caller; `depth` is what [`GraphData::filter_depth`] filters on). An edge
//! `source -> target` is directed, and its meaning is the caller's: for the updaters, a node's
//! *incoming* count is its physical mass, so `source -> target` should read as
//! "source points at / follows / belongs to target".

use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// One node of a JSON graph.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct GraphNode {
    pub id: i64,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub depth: Option<u32>,
}

/// One directed edge of a JSON graph.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct GraphEdge {
    pub source: i64,
    pub target: i64,
}

/// A whole JSON graph: nodes plus directed edges.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct GraphData {
    #[serde(default)]
    pub directed: Option<bool>,
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    Parse(serde_json::Error),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "io error: {e}"),
            LoadError::Parse(e) => write!(f, "json error: {e}"),
        }
    }
}

impl std::error::Error for LoadError {}

impl GraphData {
    /// Parse a graph from a JSON string.
    pub fn from_json_str(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }

    /// Parse a graph from any reader.
    pub fn from_reader<R: std::io::Read>(r: R) -> Result<Self, LoadError> {
        serde_json::from_reader(r).map_err(LoadError::Parse)
    }

    /// Parse a graph from a file.
    pub fn from_path<P: AsRef<Path>>(p: P) -> Result<Self, LoadError> {
        let f = std::fs::File::open(p).map_err(LoadError::Io)?;
        Self::from_reader(std::io::BufReader::new(f))
    }

    /// A map from node `id` to its index in [`GraphData::nodes`].
    pub fn id_index(&self) -> HashMap<i64, usize> {
        self.nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.id, i))
            .collect()
    }

    /// Keep only nodes whose `depth` is `<= max_depth` (a node with no `depth` is kept), and drop
    /// every edge that touches a dropped node. This is the cheap way to hand a big hierarchical
    /// graph to an `n^2` updater one hop at a time.
    pub fn filter_depth(&self, max_depth: u32) -> GraphData {
        let kept: HashMap<i64, ()> = self
            .nodes
            .iter()
            .filter(|n| n.depth.map(|d| d <= max_depth).unwrap_or(true))
            .map(|n| (n.id, ()))
            .collect();

        GraphData {
            directed: self.directed,
            nodes: self
                .nodes
                .iter()
                .filter(|n| kept.contains_key(&n.id))
                .cloned()
                .collect(),
            edges: self
                .edges
                .iter()
                .filter(|e| kept.contains_key(&e.source) && kept.contains_key(&e.target))
                .cloned()
                .collect(),
        }
    }

    /// `(nodes, edges)` counts, handy for a quick report.
    pub fn counts(&self) -> (usize, usize) {
        (self.nodes.len(), self.edges.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOC: &str = r#"{
        "directed": true,
        "nodes": [
            {"id": 0, "label": "a", "depth": 0},
            {"id": 1, "label": "b", "depth": 1},
            {"id": 2, "label": "c", "depth": 2},
            {"id": 3, "label": "d"}
        ],
        "edges": [
            {"source": 1, "target": 0},
            {"source": 2, "target": 1},
            {"source": 3, "target": 2}
        ]
    }"#;

    #[test]
    fn parses_nodes_and_edges() {
        let g = GraphData::from_json_str(DOC).unwrap();
        assert_eq!(g.counts(), (4, 3));
        assert_eq!(g.directed, Some(true));
        assert_eq!(g.nodes[1].label.as_deref(), Some("b"));
        assert_eq!(g.nodes[1].depth, Some(1));
        assert_eq!(g.nodes[3].depth, None);
    }

    #[test]
    fn filter_depth_drops_nodes_and_touching_edges() {
        let g = GraphData::from_json_str(DOC).unwrap().filter_depth(1);
        // node 2 (depth 2) is gone; node 3 has no depth so it is kept.
        assert_eq!(g.nodes.len(), 3);
        assert!(g.nodes.iter().all(|n| n.id != 2));
        // 2->1 dropped (source gone), 3->2 dropped (target gone), 1->0 kept.
        assert_eq!(g.edges, vec![GraphEdge { source: 1, target: 0 }]);
    }

    #[test]
    fn id_index_maps_ids_to_positions() {
        let g = GraphData::from_json_str(DOC).unwrap();
        let idx = g.id_index();
        assert_eq!(idx[&0], 0);
        assert_eq!(idx[&3], 3);
    }
}
