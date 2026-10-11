use super::Updater;
use crate::{NodeKey, PhysicalGraph, PhysicalNode};
use glam::Vec2;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

#[derive(Clone, Copy)]
struct Rect {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
}

impl Rect {
    fn new(min_x: f32, min_y: f32, max_x: f32, max_y: f32) -> Rect {
        Rect {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    /// Half-open containment: a point on a shared edge belongs to exactly one child.
    ///
    /// A closed containment (`v <= max`) put every body sitting on a split line into two children
    /// at once. That double-counted its mass in the force sum, and — worse — when bodies were
    /// coincident it put *every* body into all four children at every level, so the tree grew by
    /// `4^depth` until memory ran out: an unbounded field of overlapping nodes, or one collapsed
    /// onto a point by its own physics, killed the process. The root region is grown past the data
    /// so nothing can sit on its far edge and fall out of every child.
    fn contain(&self, v: Vec2) -> bool {
        self.min_x <= v.x && v.x < self.max_x && self.min_y <= v.y && v.y < self.max_y
    }
}

struct Body {
    position: Vec2,
    mass: f32,
}

struct TreeNode {
    // Surprisingly, region field is not need to read..
    // only center field is required
    // region: Rect,
    center: Vec2,
    radius: f32,
    mass: f32,
    children: Option<Vec<Arc<TreeNode>>>,
}

impl TreeNode {
    fn create_node(bodies: Vec<&Body>, region: Rect, depth: usize, max_depth: usize) -> TreeNode {
        if bodies.is_empty() {
            return TreeNode {
                center: Vec2::new(
                    (region.min_x + region.max_x) / 2.0,
                    (region.min_y + region.max_y) / 2.0,
                ),
                radius: 0.0,
                mass: 0.0,
                children: None,
            };
        }

        let position_sum = bodies
            .iter()
            .map(|&body| body.position * body.mass)
            .sum::<Vec2>();
        let mass = bodies.iter().map(|&body| body.mass).sum::<f32>();

        // A centre that is not finite means the bodies' positions are not: the containment test
        // below would then be true for all four children at once, and the tree would duplicate
        // every body into every one of them at each level until memory runs out. Fall back to the
        // region's own centre and make this node a leaf.
        let region_center = Vec2::new(
            (region.min_x + region.max_x) / 2.0,
            (region.min_y + region.max_y) / 2.0,
        );
        let center = if mass > 0.0 {
            position_sum / mass
        } else {
            region_center
        };
        let center = if center.is_finite() {
            center
        } else {
            region_center
        };

        let radius = bodies
            .iter()
            .filter(|body| body.position.is_finite())
            .map(|body| center.distance(body.position))
            .reduce(f32::max)
            .unwrap_or(0.0);

        let is_depth_over = if max_depth == 0 {
            false
        } else {
            max_depth <= depth
        };

        // Subdividing a region that is not finite is meaningless for the same reason as above.
        let subdividable = region.min_x.is_finite()
            && region.min_y.is_finite()
            && region.max_x.is_finite()
            && region.max_y.is_finite();

        let center_x = (region.min_x + region.max_x) / 2.0;
        let center_y = (region.min_y + region.max_y) / 2.0;

        // ...and so is subdividing one the floats can no longer split: when the centre coincides
        // with an edge, every body stays on the same side at every further level, so the recursion
        // would only descend to `max_depth` for nothing. Coincident bodies (a collapsed layout, a
        // degenerate seed) hit this immediately.
        let splittable = center_x > region.min_x
            && center_x < region.max_x
            && center_y > region.min_y
            && center_y < region.max_y;

        let children = if bodies.len() <= 1 || is_depth_over || !subdividable || !splittable {
            None
        } else {
            Some(
                vec![
                    Rect::new(region.min_x, region.min_y, center_x, center_y),
                    Rect::new(center_x, region.min_y, region.max_x, center_y),
                    Rect::new(region.min_x, center_y, center_x, region.max_y),
                    Rect::new(center_x, center_y, region.max_x, region.max_y),
                ]
                .into_par_iter()
                .map(|region| {
                    Arc::new(TreeNode::create_node(
                        bodies
                            .clone()
                            .into_iter()
                            .filter(|&body| region.contain(body.position))
                            .collect::<Vec<&Body>>(),
                        region,
                        depth + 1,
                        max_depth,
                    ))
                })
                .collect::<Vec<Arc<TreeNode>>>(),
            )
        };

        TreeNode {
            center,
            radius,
            mass,
            children,
        }
    }
}

pub struct QuadTreeUpdater {
    pub base_neighbor_radius: f32,
    pub neighbor_radius: f32,
    pub neighbor_space_ratio: f32,
    pub mass_amplification: f32,
    pub min_node_mass: f32,
    pub neighbor_edge_elasticity: f32,
    pub non_neighbor_repulsive_force: f32,
    pub non_neighbor_distance_softning: f32,
    pub attenuation_rate: f32,
    pub max_tree_node_radius_ratio: f32,
    /// Acceleration toward the origin, per unit of distance from it (`0.0` = none).
    ///
    /// Nothing else here pulls a node inward: the neighbour edges are local, and the bouncing force
    /// only ever pushes. Without a containment term a graph expands until its springs happen to
    /// balance the repulsion, and on a heavy-tailed graph that balance sits absurdly far out. A
    /// small value bounds the layout and keeps the whole thing on screen while it settles.
    pub containment_force: f32,
}

impl QuadTreeUpdater {
    /// The original preconfigured setting.
    ///
    /// It assumes a cloud spread over thousands of units (`node_physics` seeds its nodes within a
    /// radius of `10_000`) and mild degrees. On a graph seeded at unit scale, or one with a long
    /// tail of hubs, it tears itself apart — use [`QuadTreeUpdater::unit_scale_setting`] there.
    pub fn default_setting() -> Self {
        Self {
            base_neighbor_radius: 3.0,
            neighbor_radius: 0.5,
            neighbor_space_ratio: 0.2,
            mass_amplification: 0.25,
            min_node_mass: 1.0,
            neighbor_edge_elasticity: 20.0,
            non_neighbor_repulsive_force: 2000.0,
            non_neighbor_distance_softning: 0.1,
            attenuation_rate: 800.0,
            max_tree_node_radius_ratio: 0.5,
            containment_force: 0.0,
        }
    }

    /// A preconfigured setting for a graph seeded at **unit scale** — neighbours roughly one unit
    /// apart, as [`crate::layout::tidy_layout`] produces — and, unlike `default_setting`, for a
    /// long-tailed degree distribution.
    ///
    /// Every constant here is smaller than its `default_setting` counterpart by roughly the ratio
    /// between the two scales, and mass grows with in-degree far more slowly (`0.05` per edge
    /// against `0.25`): at unit scale the default mass already makes a hub of a few hundred
    /// followers outweigh the rest of the graph, and the repulsion — which is proportional to the
    /// product of two masses — then flings the layout apart.
    pub fn unit_scale_setting() -> Self {
        Self {
            base_neighbor_radius: 0.5,
            neighbor_radius: 0.02,
            neighbor_space_ratio: 0.2,
            mass_amplification: 0.05,
            min_node_mass: 1.0,
            neighbor_edge_elasticity: 2.0,
            non_neighbor_repulsive_force: 20.0,
            non_neighbor_distance_softning: 0.5,
            attenuation_rate: 300.0,
            max_tree_node_radius_ratio: 0.5,
            containment_force: 0.02,
        }
    }

    fn create_root_node<K: NodeKey>(&self, graph: &PhysicalGraph<K>) -> TreeNode {
        if graph.nodes.is_empty() {
            return TreeNode {
                center: Vec2::ZERO,
                radius: 0.0,
                mass: 0.0,
                children: None,
            };
        }

        let min_x = graph
            .nodes
            .values()
            .map(|node| node.position.x)
            .reduce(f32::min)
            .unwrap();
        let min_y = graph
            .nodes
            .values()
            .map(|node| node.position.y)
            .reduce(f32::min)
            .unwrap();
        let max_x = graph
            .nodes
            .values()
            .map(|node| node.position.x)
            .reduce(f32::max)
            .unwrap();
        let max_y = graph
            .nodes
            .values()
            .map(|node| node.position.y)
            .reduce(f32::max)
            .unwrap();

        let bodies = graph
            .nodes
            .values()
            .map(|node| Body {
                position: node.position,
                mass: self.get_mass(node),
            })
            .collect::<Vec<Body>>();

        // A body sitting exactly on the root's far edge would fall out of every half-open child, so
        // grow the region past the data first.
        let margin = ((max_x - min_x).max(max_y - min_y) * 1e-3).max(1e-6);

        TreeNode::create_node(
            bodies.iter().collect::<Vec<&Body>>(),
            Rect::new(min_x - margin, min_y - margin, max_x + margin, max_y + margin),
            0,
            127,
        )
    }

    fn update_node<K: NodeKey>(
        &self,
        graph: &PhysicalGraph<K>,
        node: &PhysicalNode<K>,
        tree_node: Arc<TreeNode>,
        dt: f32,
    ) -> PhysicalNode<K> {
        let mass = self.get_mass(node);
        let this_radius = self.get_neighbor_radius(node);

        let body = Body {
            position: node.position,
            mass: self.get_mass(node),
        };
        let global_force = self.calculate_tree_node_force(&body, tree_node);

        let neighbor_force = graph
            .nodes
            .values()
            .filter(|&other_node| {
                other_node.key != node.key
                    && (node.incomings.contains(&other_node.key)
                        || node.outgoings.contains(&other_node.key))
            })
            .map(|other_node| -> Vec2 {
                let other_node_mass = self.get_mass(&other_node);

                let edge_length = this_radius + self.get_neighbor_radius(&other_node);

                let min_distance = edge_length * (1.0 - self.neighbor_space_ratio / 2.0)
                    + self.base_neighbor_radius;
                let max_distance = edge_length * (1.0 + self.neighbor_space_ratio / 2.0)
                    + self.base_neighbor_radius;

                let diff = other_node.position - node.position;

                let distance_diff = if diff.length() < min_distance {
                    diff.length() - min_distance
                } else if max_distance < diff.length() {
                    diff.length() - max_distance
                } else {
                    0.0f32
                };

                self.neighbor_edge_elasticity
                    * diff.normalize_or_zero()
                    * distance_diff
                    * mass
                    * other_node_mass
            })
            .sum::<Vec2>();

        let velocity = node.position - node.prev_position;
        let acc = (neighbor_force + global_force) / mass
            + node.position * -self.containment_force
            - velocity * self.attenuation_rate;

        let next_position = 2.0 * node.position - node.prev_position + acc * dt.powi(2);

        // One divergent tick is enough to poison everything after it: a non-finite position makes
        // every quadtree containment test true, so the next frame's tree duplicates every body into
        // every child until memory runs out. Refuse the step rather than take it.
        let position = if next_position.is_finite() {
            next_position
        } else {
            node.position
        };

        PhysicalNode {
            key: node.key,
            incomings: node.incomings.iter().copied().collect::<HashSet<K>>(),
            outgoings: node.outgoings.iter().copied().collect::<HashSet<K>>(),
            position,
            prev_position: node.position,
        }
    }

    /// Calculates force from tree node.
    fn calculate_tree_node_force(&self, body: &Body, tree_node: Arc<TreeNode>) -> Vec2 {
        let distance = body.position.distance(tree_node.center);

        // It's not enough as f32 std epsilon. Just using distance softening value is more appropriate.
        if distance <= self.non_neighbor_distance_softning {
            return Vec2::ZERO;
        }

        if tree_node.radius < distance * self.max_tree_node_radius_ratio
            || tree_node.children.is_none()
        {
            let diff = tree_node.center - body.position;

            -diff.normalize_or_zero()
                * self.non_neighbor_repulsive_force
                * (body.mass * tree_node.mass)
                / (distance.powi(2) + self.non_neighbor_distance_softning)
        } else {
            tree_node
                .children
                .clone()
                .unwrap()
                .into_iter()
                .map(|child| self.calculate_tree_node_force(body, child))
                .sum::<Vec2>()
        }
    }

    fn get_mass<K: NodeKey>(&self, node: &PhysicalNode<K>) -> f32 {
        self.min_node_mass + node.incomings.len() as f32 * self.mass_amplification
    }
    fn get_neighbor_radius<K: NodeKey>(&self, node: &PhysicalNode<K>) -> f32 {
        node.incomings.len() as f32 * self.neighbor_radius
    }
}

impl Updater for QuadTreeUpdater {
    fn update<K: NodeKey>(&self, graph: &PhysicalGraph<K>, dt: f32) -> PhysicalGraph<K> {
        let tree_node_ref = Arc::new(self.create_root_node(&graph));

        PhysicalGraph::<K> {
            nodes: graph
                .nodes
                .keys()
                .copied()
                .collect::<Vec<K>>()
                .into_par_iter()
                .map(|node_key| {
                    (
                        node_key,
                        self.update_node(
                            &graph,
                            graph.nodes.get(&node_key).unwrap(),
                            Arc::clone(&tree_node_ref),
                            dt,
                        ),
                    )
                })
                .collect::<HashMap<K, PhysicalNode<K>>>(),
        }
    }
}
