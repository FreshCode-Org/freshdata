# scripts/run_mutmut_audit.py
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

TARGET_FILES = [
    "src/freshdata/schema_envelope.py",
    "src/freshdata/imputation.py",
    "src/freshdata/execution.py",
]


class ComparisonFlipMutator(ast.NodeTransformer):
    """Deterministic AST mutator flipping comparison and boolean operators one site at a time."""

    FLIPS = {
        ast.Eq: ast.NotEq,
        ast.NotEq: ast.Eq,
        ast.Lt: ast.GtE,
        ast.GtE: ast.Lt,
        ast.Gt: ast.LtE,
        ast.LtE: ast.Gt,
    }

    def __init__(self, target_idx: int) -> None:
        self.target_idx = target_idx
        self.current_idx = 0

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if len(node.ops) == 1 and type(node.ops[0]) in self.FLIPS:
            if self.current_idx == self.target_idx:
                node.ops[0] = self.FLIPS[type(node.ops[0])]()
            self.current_idx += 1
        return node


def count_sites(tree: ast.AST) -> int:
    mut = ComparisonFlipMutator(-1)
    mut.visit(tree)
    return mut.current_idx


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    killed = 0
    survived = 0

    for rel_path in TARGET_FILES:
        fpath = repo_root / rel_path
        orig_src = fpath.read_text(encoding="utf-8")
        n_sites = min(count_sites(ast.parse(orig_src)), 15)

        for idx in range(n_sites):
            tree = ast.parse(orig_src)
            mutated_tree = ComparisonFlipMutator(idx).visit(tree)
            ast.fix_missing_locations(mutated_tree)
            fpath.write_text(ast.unparse(mutated_tree), encoding="utf-8")

            res = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "--tb=no"],
                cwd=repo_root,
                capture_output=True,
                text=True,
            )
            if res.returncode != 0:
                killed += 1
            else:
                survived += 1
                print(f"[SURVIVED] {rel_path} mutant #{idx}")
        fpath.write_text(orig_src, encoding="utf-8")

    total = killed + survived
    score = (killed / total * 100.0) if total else 100.0
    print(f"MUTATION_AUDIT_SUMMARY: killed={killed}/{total} ({score:.1f}%)")
    return 0 if score >= 85.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
