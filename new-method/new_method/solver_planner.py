"""Deterministic solver-policy selection for Adaptive Computation Graphs.

The planner intentionally reasons from graph structure and metadata. It does not
branch on benchmark names or require a closed taxonomy of school-math topics.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from typing import Any, Mapping


FUNCTION_NAMES = {
    "abs", "bin", "comb", "cos", "exp", "factorial", "gcd", "int", "lcm",
    "log", "max", "min", "oct", "perm", "pi", "sin", "sqrt", "tan", "Tuple",
}


def _symbols(text: Any) -> set[str]:
    return {
        name for name in re.findall(r"\b[A-Za-z_]\w*\b", str(text or ""))
        if name not in FUNCTION_NAMES and name not in {"True", "False", "and", "or", "not"}
    }


def _is_definition(edge: Mapping[str, Any]) -> bool:
    # The graph kind is authoritative: a constraint may use `=` while still
    # requiring solving rather than forward evaluation.
    return str(edge.get("kind")) == "definition"


def _acyclic_definitions(edges: list[Mapping[str, Any]]) -> bool:
    graph: dict[str, set[str]] = defaultdict(set)
    indegree: dict[str, int] = defaultdict(int)
    nodes: set[str] = set()
    for edge in edges:
        lhs = str(edge.get("lhs") or "").strip()
        if not re.fullmatch(r"[A-Za-z_]\w*", lhs):
            continue
        nodes.add(lhs)
        for source in _symbols(edge.get("rhs")):
            nodes.add(source)
            if lhs not in graph[source]:
                graph[source].add(lhs)
                indegree[lhs] += 1
    queue = deque(node for node in nodes if indegree[node] == 0)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for child in graph[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return visited == len(nodes)


def _finite_domain(graph: Mapping[str, Any]) -> bool:
    for edge in graph.get("edges", []):
        if isinstance(edge, Mapping) and edge.get("range"):
            return True
    for condition in graph.get("conditions", []):
        text = str(condition.get("expr") if isinstance(condition, Mapping) else condition).lower()
        if "integer" in text and any(token in text for token in ("<", ">", "range", "between")):
            return True
    return False


def _tags(graph: Mapping[str, Any]) -> set[str]:
    tags: set[str] = set()
    for edge in graph.get("edges", []):
        if not isinstance(edge, Mapping):
            continue
        tags.update(str(tag).lower() for tag in (edge.get("tags") or []))
        tags.add(str(edge.get("intent") or "").lower())
        tags.add(str(edge.get("operation") or "").lower())
    return tags


def plan_solver(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-friendly plan with a conservative fallback strategy."""
    edges = [edge for edge in graph.get("edges", []) if isinstance(edge, Mapping)]
    executable = [edge for edge in edges if edge.get("executable", True) is not False]
    tags = _tags(graph)
    has_constraints = any(not _is_definition(edge) or edge.get("kind") in {"constraint", "selection", "verification"} for edge in executable)
    has_selection = any(edge.get("kind") == "selection" for edge in executable)
    has_combinatorics = bool(tags & {"combinatorics", "count", "counting", "probability", "comb", "perm", "factorial"})
    has_geometry = bool(tags & {"geometry", "distance", "area", "volume", "point", "segment"})
    has_symbolic = any(any(token in str(edge.get("rhs", "")) for token in ("sqrt", "solve", "factor", "expand", "pi", "I", "Tuple")) for edge in executable)
    finite_domain = _finite_domain(graph)
    acyclic = _acyclic_definitions(executable)

    if has_selection or finite_domain:
        strategy = "enumerative_search"
    elif has_combinatorics and not has_constraints:
        strategy = "combinatorics_formula"
    elif has_geometry and not has_constraints and acyclic:
        strategy = "geometry_helper"
    elif has_constraints and any(_is_definition(edge) for edge in executable):
        strategy = "hybrid"
    elif has_constraints or has_symbolic:
        strategy = "symbolic_solve"
    elif executable and acyclic:
        strategy = "sequential_eval"
    else:
        strategy = "hybrid"

    ordered_steps = []
    for edge in executable:
        action = {
            "sequential_eval": "evaluate_definition",
            "symbolic_solve": "solve_relation",
            "enumerative_search": "enumerate_candidates",
            "combinatorics_formula": "evaluate_counting_operation",
            "geometry_helper": "evaluate_geometry_relation",
            "hybrid": "evaluate_or_solve",
        }[strategy]
        if edge.get("kind") in {"constraint", "selection", "verification"}:
            action = "check_constraint" if edge.get("kind") != "selection" else "select_candidates"
        ordered_steps.append({"edge_id": edge.get("id"), "action": action})

    risks: list[str] = []
    if not executable:
        risks.append("no executable edges")
    if not acyclic and strategy == "sequential_eval":
        risks.append("definition dependencies are cyclic")
    if any(not edge.get("source") or not edge.get("evidence") for edge in edges):
        risks.append("one or more edges lack provenance")
    if any(edge.get("unit") for edge in edges) and any("unit" in str(tag) for tag in tags):
        risks.append("unit semantics require verification")

    fallbacks = [item for item in ("hybrid", "symbolic_solve", "enumerative_search") if item != strategy]
    return {
        "version": "acg-plan-v1",
        "strategy": strategy,
        "signals": {
            "acyclic_definitions": acyclic,
            "has_constraints": has_constraints,
            "has_selection": has_selection,
            "finite_domain": finite_domain,
            "has_combinatorics": has_combinatorics,
            "has_geometry": has_geometry,
            "has_symbolic": has_symbolic,
        },
        "ordered_steps": ordered_steps,
        "required_libraries": ["sympy", "math", "fractions"],
        "risk_flags": risks,
        "fallbacks": fallbacks,
    }
