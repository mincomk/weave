# weave
Simple physics engine for node visualization

# dependency
```toml
weave = { git = "https://github.com/ityeri/weave", tag = "v0.1.0" }
```

# examples
* `cargo run -r --example file_tree -- /some/directory -w 0.3` : runs a file tree visualizer example. `-w` is a wheel sensitivity option. (0.3 recommended)
* `cargo run -r --example node_physics` : runs an example which simulates two central nodes and sub nodes of them.
* `cargo run -r --example json_graph -- graph.json -w 0.3 --depth 2` : visualises a graph loaded from a JSON file.
* `cargo run -r --example json_stats -- graph.json --depth 2 --ticks 300` : loads the same file headlessly, runs the physics and reports tick time, layout radius and RSS.

# JSON graphs
The `json` feature (on by default) adds `weave::json_loader`, a small serde-only bridge from a JSON
file to a graph. The wire format is generic — any node/edge graph, dumped from any language:

```json
{
  "directed": true,
  "nodes": [
    { "id": 0, "label": "ityeri", "depth": 0 },
    { "id": 1, "label": "Jung2010", "depth": 1 }
  ],
  "edges": [
    { "source": 1, "target": 0 }
  ]
}
```

`id` is a stable integer identity; `label` and `depth` are optional metadata (`depth` is what
`GraphData::filter_depth(max)` filters on, the cheap way to feed a big hierarchical graph to an
`n^2` updater one hop at a time). An edge `source -> target` is directed and its meaning is the
caller's — the updaters treat a node's *incoming* count as its physical mass, so an edge should read
as "source points at / follows / belongs to target".

```rust
use weave::json_loader::GraphData;

let data = GraphData::from_path("graph.json")?;
let data = data.filter_depth(2);           // keep hops 0..=2
let (nodes, edges) = data.counts();
```

`weave::layout` seeds a graph so the physics has somewhere to start: `seed_layout(&data)` gives one
ring per hop — `GraphData::depth` when the file has it, BFS depth otherwise — with the nodes on each
ring ordered by the mean angle of their neighbours on the ring inside, so the first frame is
structure rather than confetti. `adjacency_of`, `depths_of` and `tidy_layout` are the pieces.

`examples/json_graph.rs` is the consumer to copy from: it loads the file, maps each node `id` to a
`petgraph::StableGraph` node, seeds it with `weave::layout`, and drives it with `QuadTreeUpdater` and
the same `scrollrs` camera as `node_physics`. `examples/json_stats.rs` does the same load and runs
the physics without a window, reporting tick time, layout radius, RSS and the farthest nodes — how a
slice or a set of updater constants is checked without watching a screen.

`tools/to_weave.py` converts a login-keyed follow-graph dump (`{"nodes":[{"login",...}], "edges":[{"source","target"}]}`)
into this format, and `tools/sample-graph.json` is a small graph to try the examples on.

> The neighbour pass is currently `O(n^2)`, so a few thousand nodes is the practical ceiling for an
> interactive session (~2 ms/tick at 640 nodes, ~2.5 s/tick at 40 587 nodes on a laptop-class CPU).
> `json_graph` therefore starts with the physics frozen above `--max-auto-nodes` (5000) and with the
> edges hidden above `--max-edge-nodes` (20 000).

# Physics
`QuadTreeUpdater::default_setting()` is tuned for `node_physics`'s cloud — nodes seeded within a
radius of `10 000` with small degrees. On a graph seeded at unit scale, and especially on one with a
long tail of hubs, it takes itself apart: at 50 ticks a 640-node follow graph is at a median radius
of `1.4e18`, because the bouncing force is proportional to the *product* of two masses and mass grows
with in-degree while nothing at all pulls inward.

Use `QuadTreeUpdater::unit_scale_setting()` for a graph seeded by `weave::layout` — it is the same
model with every constant scaled to a unit-scale seed, slower mass growth, and a `containment_force`
(a pull toward the origin) so the layout is bounded. The same 640-node graph settles at a stable
median radius of ~12 units and stays there over thousands of ticks.

Two things were also fixed in the tree itself, both of which could take the process down:

* **Overlapping children.** A body exactly on a split line was contained by two children at once
  (closed `<=` on both edges), so its mass was counted twice in the force sum — and when bodies were
  coincident, *every* body went into all four children at every level and the tree grew as `4^depth`
  until memory ran out. Containment is now half-open, and the root region is grown past the data so
  nothing can sit on its far edge.
* **Non-finite and degenerate state.** A divergent tick used to be able to write an infinite or `NaN`
  position, after which every containment test is true for every child: same explosion. A step whose
  result is not finite is now refused, a region that is not finite (or that the floats can no longer
  split) is treated as a leaf, and an empty graph returns an empty tree instead of panicking.


# `Updater`
All updaters follow this trait: `Updater`.
Simply, it gets the previous data of `PhysicalGraph` and time delta **secondes**, and then returns the next frame of it.

```
fn update<K: NodeKey>(&self, graph: &PhysicalGraph<K>, dt: f32) -> PhysicalGraph<K>;
```

usage
```rust
use weave::updater::{SomeUpdaterImpl, Updater};

let updater: Updater = SomeUpdaterImpl::new();

let updated_graph = updater.update(physical_graph, 1.0 / 60.0);
```

## `DefaultUpdater`
Most simple, almost non-optimized except for multi-threading.
`DefaultUpdater::default_setting()` is a preconfigured setting which is appropriate for most of general case.
Time complexity is `n^2`.

The number of incoming edges is treated as physical mass.
If two nodes are connected through the edge, they are connected with a physical edge which length is proportion with sum of each node's incomings.
If two nodes are non-neighbor, it pushes back on each other. (opposite of the general gravity force)

usage
```rust
use weave::updater::{DefaultUpdater, Updater};

let updater: Updater = DefaultUpdater::default_setting();

let updated_graph = updater.update(physical_graph, 1.0 / 60.0);
```


## `QuadTreeUpdater`
Optimized with the barnes-hut algorithm in the bouncing force part.

Basically, most of part is same as `DefaultUpdater`, but all nodes push back on each other even if they are neighbor.
If they are neighbor, physical edges are graps node can't go too far.

usage
```rust
use weave::updater::{QuadTreeUpdater, Updater};

let updater: Updater = QuadTreeUpdater::default_setting();

let updated_graph = updater.update(physical_graph, 1.0 / 60.0);
```

# to do
This project still has a long way to go.
It's developing for a million-to-billion scale node physics, but it's not enough to optimize for it.
Most development of this project will be optimization.

* GPGPU - wgpu or rust-gpu
* Dynamic grid
* Hm

