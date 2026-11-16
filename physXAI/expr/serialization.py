"""
Json serialization of expression graphs.

A graph is stored as a list of nodes in topological order, every node referring to its
children by id. Shared sub-graphs (e.g. a parameter used twice) stay shared after loading.

    {"nodes": [{"id": "n0", "type": "Var", "name": null, "params": {"column": "T"}, "inputs": []}, ...],
     "outputs": ["n3"]}
"""

from __future__ import annotations

from physXAI.expr.core import EXPRESSION_REGISTRY, Expr


def to_config(outputs: Expr | list[Expr]) -> dict:
    """Serializes one expression or a list of expressions (sharing nodes) into a json dict."""
    single = isinstance(outputs, Expr)
    outputs = [outputs] if single else list(outputs)
    ids: dict[int, str] = dict()
    nodes = list()
    for output in outputs:
        for node in output.nodes():
            if id(node) in ids:
                continue
            ids[id(node)] = f"n{len(ids)}"
            nodes.append(
                {
                    "id": ids[id(node)],
                    "type": node.__class__.__name__,
                    "name": node._name,
                    "params": node.params(),
                    "inputs": [ids[id(c)] for c in node.children],
                }
            )
    return {"nodes": nodes, "outputs": [ids[id(o)] for o in outputs], "single": single}


def from_config(config: dict) -> Expr | list[Expr]:
    """Restores the expression(s) serialized by `to_config`."""
    built: dict[str, Expr] = dict()
    for node in config["nodes"]:
        try:
            cls = EXPRESSION_REGISTRY[node["type"]]
        except KeyError:
            raise ValueError(f"Unknown expression type '{node['type']}'.") from None
        children = [built[i] for i in node["inputs"]]
        params = {k: tuple(v) if k == "nominal_range" and v is not None else v for k, v in node["params"].items()}
        built[node["id"]] = cls.from_params(params, children, name=node["name"])
    outputs = [built[i] for i in config["outputs"]]
    return outputs[0] if config.get("single", len(outputs) == 1) else outputs
