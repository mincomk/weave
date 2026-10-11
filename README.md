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
* `cargo run -r --example json_stats -- graph.json --depth 2` : loads the same file headlessly and times one update tick.

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

`examples/json_graph.rs` is the consumer to copy from: it loads the file, maps each node `id` to a
`petgraph::StableGraph` node, and drives it with `QuadTreeUpdater` and the same `scrollrs` camera as
`node_physics` (drag to pan, wheel to zoom, space to toggle the physics, `f` to fit the layout to
the window). `examples/json_stats.rs` runs the same load and one tick without a window.

`tools/to_weave.py` converts a login-keyed follow-graph dump (`{"nodes":[{"login",...}], "edges":[{"source","target"}]}`)
into this format, and `tools/sample-graph.json` is a small graph to try the examples on.

> The neighbour pass is currently `O(n^2)`, so a few thousand nodes is the practical ceiling for an
> interactive session (~2 ms/tick at 640 nodes, ~2.5 s/tick at 40 587 nodes on a laptop-class CPU).

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

