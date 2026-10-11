#!/usr/bin/env python3
"""Convert a login-keyed follow-graph dump into weave's JSON graph format.

Input : {"nodes": [{"login", "depth"}, ...],
         "edges": [{"source": <login>, "target": <login>}, ...], ...}
        Edge A -> B means "A follows B".
Output: {"directed": true,
         "nodes": [{"id": int, "label": login, "depth": int}, ...],
         "edges": [{"source": int, "target": int}, ...]}

The node `id` is the index of the node in the input node list. Edges are kept only when both
endpoints exist as nodes (dangling endpoints and self-loops are counted and reported, never
emitted). Because the updaters treat a node's *incoming* count as its mass, keep the direction
"follower -> followee" so that a popular account is heavy.

Usage: to_weave.py follow-graph-data.json follow-graph-weave.json
"""
import json
import sys


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else "follow-graph-data.json"
    dst = sys.argv[2] if len(sys.argv) > 2 else "follow-graph-weave.json"

    with open(src) as f:
        raw = json.load(f)

    id_of = {}
    nodes = []
    for i, n in enumerate(raw["nodes"]):
        login = n["login"]
        id_of[login] = i
        node = {"id": i, "label": login}
        if n.get("depth") is not None:
            node["depth"] = n["depth"]
        nodes.append(node)

    edges = []
    dangling = 0
    self_loops = 0
    for e in raw["edges"]:
        s = id_of.get(e["source"])
        t = id_of.get(e["target"])
        if s is None or t is None:
            dangling += 1
            continue
        if s == t:
            self_loops += 1
            continue
        edges.append({"source": s, "target": t})

    with open(dst, "w") as f:
        json.dump({"directed": True, "nodes": nodes, "edges": edges}, f,
                  separators=(",", ":"), ensure_ascii=False)

    depths = {}
    for n in nodes:
        depths[n.get("depth")] = depths.get(n.get("depth"), 0) + 1

    print(f"{dst}")
    print(f"  nodes : {len(nodes)}")
    print(f"  edges : {len(edges)}  (raw {len(raw['edges'])}, dangling {dangling}, self-loops {self_loops})")
    print(f"  by depth: {dict(sorted(depths.items(), key=lambda kv: (kv[0] is None, kv[0])))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
