from __future__ import annotations

import argparse
import difflib
import sys
from collections.abc import Sequence
from pathlib import Path

import libcst as cst
from libcst.codemod import (
    CodemodContext,
    TransformSuccess,
    VisitorBasedCodemodCommand,
    transform_module,
)
from libcst.codemod.visitors import AddImportsVisitor

KWARG_RENAMES: dict[str, str] = {
    "impute": "default_impute",
    "outliers": "default_outliers",
    "column_names": "normalize_names",
}

REWRITTEN_TO_CLEAN: frozenset[str] = frozenset(
    {
        "clean_csv",
        "clean_excel",
        "fill_missing",
        "remove_outliers",
        "resolve_duplicates",
        "pipeline_clean",
    }
)

MANUAL_MIGRATION_GUIDANCE: dict[str, str] = {
    "compile_context": "Replace natural-language rules with explicit fd.ColumnRule objects in fd.CleanConfig(rules=...).",
    "compute_trust_score": "Removed in v3.0. Inspect fd.profile(df) or CleanReport audit metrics directly.",
    "generate_compliance_report": "Removed in v3.0. Export CleanReport JSON to your external governance pipeline.",
    "parse_domain": "Removed in v3.0. Parse domain files with specialized format libraries before calling fd.clean().",
    "clean_domain_file": "Removed in v3.0. Parse domain files with specialized format libraries before calling fd.clean().",
}


class FreshDataV3Command(VisitorBasedCodemodCommand):
    DESCRIPTION: str = "Rewrites deprecated freshdata 2.x calls into the 3.0 core API."

    def __init__(self, context: CodemodContext) -> None:
        super().__init__(context)
        self._module_aliases: set[str] = set()
        self._direct_imports: dict[str, str] = {}

    def visit_Import(self, node: cst.Import) -> None:  # noqa: N802
        for alias in node.names:
            if isinstance(alias.name, cst.Name) and alias.name.value == "freshdata":
                asname = (
                    alias.asname.name.value
                    if alias.asname and isinstance(alias.asname.name, cst.Name)
                    else "freshdata"
                )
                self._module_aliases.add(asname)

    def leave_ImportFrom(  # noqa: N802
        self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom
    ) -> cst.ImportFrom | cst.RemovalSentinel:
        if not isinstance(updated_node.module, cst.Name) or updated_node.module.value != "freshdata":
            return updated_node
        if isinstance(updated_node.names, cst.ImportStar):
            return updated_node

        new_aliases: list[cst.ImportAlias] = []
        seen_names: set[str] = set()

        for alias in updated_node.names:
            if not isinstance(alias.name, cst.Name):
                new_aliases.append(alias)
                continue

            orig_name = alias.name.value
            local_name = (
                alias.asname.name.value
                if alias.asname and isinstance(alias.asname.name, cst.Name)
                else orig_name
            )
            self._direct_imports[local_name] = orig_name

            if orig_name == "compare_plans":
                continue

            target_name = "clean" if orig_name in REWRITTEN_TO_CLEAN else orig_name
            if target_name in seen_names:
                continue
            seen_names.add(target_name)

            if target_name != orig_name:
                new_aliases.append(cst.ImportAlias(name=cst.Name(target_name)))
            else:
                new_aliases.append(alias)

        if not new_aliases:
            return cst.RemoveFromParent()

        return updated_node.with_changes(names=new_aliases)

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.BaseExpression:  # noqa: N802
        resolved = self._resolve_freshdata_target(updated_node.func)
        if resolved is None:
            return updated_node

        prefix, func_name = resolved

        if func_name in MANUAL_MIGRATION_GUIDANCE:
            self.warn(f"'{func_name}' requires manual migration: {MANUAL_MIGRATION_GUIDANCE[func_name]}")
            return updated_node

        if func_name == "compare_plans" and len(updated_node.args) == 2:
            left_arg = updated_node.args[0].value
            right_arg = updated_node.args[1].value
            return cst.Call(
                func=cst.Attribute(value=left_arg, attr=cst.Name("diff")),
                args=[cst.Arg(value=right_arg)],
            )

        clean_callee = self._build_callee(prefix, "clean")

        if func_name in ("clean_csv", "clean_excel") and updated_node.args:
            AddImportsVisitor.add_needed_import(self.context, "pandas", asname="pd")
            reader_name = "read_csv" if func_name == "clean_csv" else "read_excel"
            path_arg = updated_node.args[0]
            pd_read_call = cst.Call(
                func=cst.Attribute(value=cst.Name("pd"), attr=cst.Name(reader_name)),
                args=[path_arg.with_changes(comma=cst.MaybeSentinel.DEFAULT)],
            )
            remaining_args = self._rename_kwargs(updated_node.args[1:])
            new_args = [cst.Arg(value=pd_read_call), *remaining_args]
            return updated_node.with_changes(func=clean_callee, args=new_args)

        if func_name == "fill_missing" and updated_node.args:
            df_arg = updated_node.args[0].with_changes(comma=cst.MaybeSentinel.DEFAULT)
            method_val: cst.BaseExpression = cst.SimpleString('"auto"')
            for arg in updated_node.args[1:]:
                if arg.keyword and arg.keyword.value == "method":
                    method_val = arg.value
            return updated_node.with_changes(
                func=clean_callee,
                args=[
                    df_arg,
                    cst.Arg(
                        keyword=cst.Name("default_impute"),
                        value=method_val,
                        equal=cst.AssignEqual(
                            whitespace_before=cst.SimpleWhitespace(""),
                            whitespace_after=cst.SimpleWhitespace(""),
                        ),
                    ),
                ],
            )

        if func_name == "remove_outliers" and updated_node.args:
            df_arg = updated_node.args[0].with_changes(comma=cst.MaybeSentinel.DEFAULT)
            return updated_node.with_changes(
                func=clean_callee,
                args=[
                    df_arg,
                    cst.Arg(
                        keyword=cst.Name("default_outliers"),
                        value=cst.SimpleString('"drop"'),
                        equal=cst.AssignEqual(
                            whitespace_before=cst.SimpleWhitespace(""),
                            whitespace_after=cst.SimpleWhitespace(""),
                        ),
                    ),
                ],
            )

        if func_name == "resolve_duplicates" and updated_node.args:
            df_arg = updated_node.args[0].with_changes(comma=cst.MaybeSentinel.DEFAULT)
            return updated_node.with_changes(
                func=clean_callee,
                args=[
                    df_arg,
                    cst.Arg(
                        keyword=cst.Name("drop_duplicates"),
                        value=cst.Name("True"),
                        equal=cst.AssignEqual(
                            whitespace_before=cst.SimpleWhitespace(""),
                            whitespace_after=cst.SimpleWhitespace(""),
                        ),
                    ),
                ],
            )

        if func_name == "pipeline_clean":
            return updated_node.with_changes(
                func=clean_callee,
                args=self._rename_kwargs(updated_node.args),
            )

        if func_name in ("clean", "CleanConfig", "Cleaner"):
            return updated_node.with_changes(args=self._rename_kwargs(updated_node.args))

        return updated_node

    def _resolve_freshdata_target(
        self, func: cst.BaseExpression
    ) -> tuple[str | None, str] | None:
        if (
            isinstance(func, cst.Attribute)
            and isinstance(func.value, cst.Name)
            and func.value.value in self._module_aliases
        ):
            return func.value.value, func.attr.value
        if isinstance(func, cst.Name) and func.value in self._direct_imports:
            return None, self._direct_imports[func.value]
        return None

    @staticmethod
    def _build_callee(prefix: str | None, name: str) -> cst.BaseExpression:
        if prefix is None:
            return cst.Name(name)
        return cst.Attribute(value=cst.Name(prefix), attr=cst.Name(name))

    @staticmethod
    def _rename_kwargs(args: Sequence[cst.Arg]) -> list[cst.Arg]:
        updated: list[cst.Arg] = []
        for arg in args:
            if arg.keyword and arg.keyword.value in KWARG_RENAMES:
                new_kw = cst.Name(KWARG_RENAMES[arg.keyword.value])
                updated.append(arg.with_changes(keyword=new_kw))
            else:
                updated.append(arg)
        return updated


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="freshdata migrate",
        description="Migrate deprecated freshdata 2.x calls to the 3.0 core API.",
    )
    parser.add_argument("paths", nargs="+", type=Path, help="Files or directories to migrate.")
    parser.add_argument("--check", action="store_true", help="Exit non-zero if files need migration.")
    parser.add_argument("--diff", action="store_true", help="Print unified diff without modifying files.")
    args = parser.parse_args(argv)

    py_files: list[Path] = []
    for p in args.paths:
        if p.is_dir():
            py_files.extend(sorted(p.rglob("*.py")))
        elif p.is_file():
            py_files.append(p)

    needs_changes = False
    for file_path in py_files:
        source = file_path.read_text(encoding="utf-8")
        context = CodemodContext(filename=str(file_path))
        command = FreshDataV3Command(context)
        result = transform_module(command, source)

        for warning in context.warnings:
            print(f"[WARN] {file_path}: {warning}", file=sys.stderr)

        if isinstance(result, TransformSuccess) and result.code != source:
            needs_changes = True
            if args.diff or args.check:
                diff_lines = difflib.unified_diff(
                    source.splitlines(keepends=True),
                    result.code.splitlines(keepends=True),
                    fromfile=f"a/{file_path}",
                    tofile=f"b/{file_path}",
                )
                sys.stdout.writelines(diff_lines)
            else:
                file_path.write_text(result.code, encoding="utf-8")
                print(f"Migrated {file_path}")

    return 1 if (args.check and needs_changes) else 0


if __name__ == "__main__":
    raise SystemExit(main())
