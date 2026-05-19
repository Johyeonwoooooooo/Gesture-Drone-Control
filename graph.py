"""
graph.py
========
Room-level graph representation and DFS-based path search.

The graph is a plain undirected adjacency list (``dict[str, list[str]]``).
``dfs_path`` returns *one* valid room sequence from *start* to *goal*;
``all_paths`` returns every simple path (useful for comparing alternatives).
"""

from __future__ import annotations

from collections import deque
from typing import Optional


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def build_graph(adjacency: dict[str, list[str]]) -> dict[str, set[str]]:
    """
    Convert a directed adjacency list to an undirected ``{node: set(neighbours)}``
    graph, ensuring symmetry.
    """
    graph: dict[str, set[str]] = {node: set(neighbours) for node, neighbours in adjacency.items()}
    for node, neighbours in list(graph.items()):
        for nb in neighbours:
            graph.setdefault(nb, set()).add(node)
    return graph


# ---------------------------------------------------------------------------
# DFS – one shortest path (BFS under the hood for optimality)
# ---------------------------------------------------------------------------

def dfs_path(
    graph: dict[str, set[str]],
    start: str,
    goal: str,
) -> Optional[list[str]]:
    """
    Return a room sequence ``[start, ..., goal]`` using iterative DFS.

    Returns the *first* path found (not necessarily shortest); for the
    guaranteed shortest hop-count path use :func:`bfs_path`.

    Returns ``None`` if no path exists.
    """
    if start not in graph:
        raise KeyError(f"Start room {start!r} not in graph.")
    if goal not in graph:
        raise KeyError(f"Goal room {goal!r} not in graph.")
    if start == goal:
        return [start]

    stack: list[tuple[str, list[str]]] = [(start, [start])]
    while stack:
        node, path = stack.pop()
        for neighbour in graph[node]:
            if neighbour in path:           # avoid cycles
                continue
            new_path = path + [neighbour]
            if neighbour == goal:
                return new_path
            stack.append((neighbour, new_path))
    return None


def bfs_path(
    graph: dict[str, set[str]],
    start: str,
    goal: str,
) -> Optional[list[str]]:
    """
    Return the *shortest* (fewest hops) room sequence via BFS.

    Preferred over :func:`dfs_path` for production use; DFS is kept for
    reference and benchmarking.
    """
    if start == goal:
        return [start]
    visited = {start}
    queue: deque[list[str]] = deque([[start]])
    while queue:
        path = queue.popleft()
        for neighbour in graph[path[-1]]:
            if neighbour in visited:
                continue
            new_path = path + [neighbour]
            if neighbour == goal:
                return new_path
            visited.add(neighbour)
            queue.append(new_path)
    return None


def all_paths(
    graph: dict[str, set[str]],
    start: str,
    goal: str,
    max_paths: int = 50,
) -> list[list[str]]:
    """
    Enumerate all simple paths from *start* to *goal* (up to *max_paths*).

    Useful for comparing alternative routing strategies.
    """
    results: list[list[str]] = []
    stack: list[tuple[str, list[str]]] = [(start, [start])]
    while stack and len(results) < max_paths:
        node, path = stack.pop()
        for neighbour in graph[node]:
            if neighbour in path:
                continue
            new_path = path + [neighbour]
            if neighbour == goal:
                results.append(new_path)
            else:
                stack.append((neighbour, new_path))
    return results


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def describe_path(room_sequence: list[str]) -> str:
    """Human-readable one-liner for a room sequence."""
    return " → ".join(room_sequence) + f"  ({len(room_sequence)} rooms)"
