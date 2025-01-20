"""
Test Runner for Python.
"""
from ast import (
    NodeVisitor,
    ClassDef,
    FunctionDef,
    AsyncFunctionDef,
    parse,
    For,
    While,
    With,
    If
)
import ast
from pathlib import Path
from typing import Dict, overload

from .data import Hierarchy, TestInfo

# pylint: disable=invalid-name, no-self-use


class TestOrder(NodeVisitor):
    """
    Visits test_* methods in a file and caches their definition order.
    """

    _cache: Dict[Hierarchy, TestInfo] = {}

    def __init__(self, root: Hierarchy) -> None:
        super().__init__()
        self._hierarchy = [root]

    def visit_ClassDef(self, node: ClassDef) -> None:
        """
        Handles class definitions.
        """
        bases = {f"{base.value.id}.{base.attr}" for base in node.bases}

        if "unittest.TestCase" in bases:
            self._hierarchy.append(Hierarchy(node.name))

        self.generic_visit(node)
        self._hierarchy.pop()

    @overload
    def _visit_definition(self, node: FunctionDef) -> None:
        ...

    @overload
    def _visit_definition(self, node: AsyncFunctionDef) -> None:
        ...

    def _visit_definition(self, node):
        if node.name.startswith("test_"):
            last_body = node.body[-1]

            # We need to account for subtests here by including "With" nodes
            while isinstance(last_body, (For, While, If, With)):
                last_body = last_body.body[-1]

            testinfo = TestInfo(node.lineno, last_body.lineno, 1)
            base_test_id = self.get_hierarchy(Hierarchy(node.name))
            self._cache[base_test_id] = testinfo

            # Add entries for parameterized variants if needed
            self._add_parameterized_variants(node, base_test_id, testinfo)

        self.generic_visit(node)

    def visit_FunctionDef(self, node: FunctionDef) -> None:
        """
        Handles test definitions
        """
        self._visit_definition(node)

    def visit_AsyncFunctionDef(self, node: AsyncFunctionDef) -> None:
        """
        Handles async test definitions
        """
        self._visit_definition(node)

    def get_hierarchy(self, name: Hierarchy) -> Hierarchy:
        """
        Returns the hierarchy :: joined.
        """
        return Hierarchy("::".join(self._hierarchy + [name]))

    def _add_parameterized_variants(self, node, base_test_id: Hierarchy,
                                    testinfo: TestInfo):
        """
        Checks for pytest.mark.parametrize decorators and adds
        parameterized variants.
        """
        for decorator in node.decorator_list:
            if (decorator.func.value.id == "pytest" and 
                    decorator.func.attr == "mark.parametrize"):
                variants = self._extract_parameters(decorator)
                for i, variant in enumerate(variants):
                    # Generate a parameterized test ID
                    parameterized_test_id = f"{base_test_id}[{variant}]"
                    self._cache[parameterized_test_id] = testinfo

    def _extract_parameters(self, decorator: ast.Call) -> list[str]:
        """
        Extracts parameterized values from the pytest.mark.parametrize
        decorator.
        """
        if len(decorator.args) < 2:
            return []

        param_values = decorator.args[1]
        if isinstance(param_values, ast.List):
            return [ast.unparse(value) for value in param_values.elts]
        return []

    @classmethod
    def lineno(cls, test_id: Hierarchy, source: Path) -> int:
        """
        Returns the line that the given test was defined on.
        """
        # Normalize test ID for parameterized tests
        normalized_test_id = cls._normalize_test_id(test_id)

        if normalized_test_id not in cls._cache:
            tree = parse(source.read_text(), source.name)
            cls(Hierarchy(test_id.split("::")[0])).visit(tree)
        print(f"cls._cache: {cls._cache}")
        return cls._cache[normalized_test_id].lineno

    @staticmethod
    def _normalize_test_id(test_id: Hierarchy) -> Hierarchy:
        """
        Removes parameterized suffixes from the test ID.
        """
        if "[" in test_id:
            return test_id.split("[")[0]
        return test_id

    @classmethod
    def function_source(cls, test_id: Hierarchy, source: Path) -> str:
        """

        :param test_id: Hierarchy position of test in AST
        :param source: Path of source code file
        :return: str of the source code of the given test.
        """
        text = source.read_text()
        testinfo = cls._cache[test_id]

        lines = text.splitlines()[testinfo.lineno: testinfo.end_lineno + 1]

        if test_id not in cls._cache:
            tree = parse(text, source.name)
            cls(Hierarchy(test_id.split("::")[0])).visit(tree)

        if not lines[-1]:
            lines.pop()

        # Dedents source.
        while all(line.startswith(' ') for line in lines if line):
            lines = [line[1:] if line else line for line in lines]
        return '\n'.join(lines)
