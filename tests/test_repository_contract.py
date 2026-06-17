from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


class RepositoryContractTests(unittest.TestCase):
    def test_methodology_overview_covers_required_sections(self) -> None:
        text = read_text("docs/METHODOLOGY_OVERVIEW.md")
        for heading in [
            "## High-Level Pipeline",
            "## Dataset Description and Access",
            "## Cleaning, Preprocessing, Transformation, and Splitting",
            "## Implemented Machine-Learning Models",
            "## Evaluation Strategy and Error Analysis",
            "## Reproducibility Commands",
        ]:
            self.assertIn(heading, text)

    def test_readme_exposes_crd_entry_points(self) -> None:
        text = read_text("README.md")
        self.assertIn("Code Repository Deliverable", text)
        self.assertIn("docs/METHODOLOGY_OVERVIEW.md", text)
        self.assertIn("make check", text)

    def test_config_declares_existing_pipeline_scripts(self) -> None:
        config_path = ROOT / "src" / "config.py"
        tree = ast.parse(config_path.read_text(encoding="utf-8"))
        assignments = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value

        eda_scripts = ast.literal_eval(assignments["EDA_SCRIPTS"])
        grouped_scripts = ast.literal_eval(assignments["GROUPED_EDA_SCRIPTS"])

        self.assertEqual(len(eda_scripts), 11)
        self.assertEqual(len(grouped_scripts), 5)
        for script in [*eda_scripts, *grouped_scripts, "src/clean_data.py", "src/run_pipeline.py"]:
            self.assertTrue((ROOT / script).exists(), script)

    def test_wordpress_runner_keeps_no_network_smoke_task(self) -> None:
        text = read_text("thesis_findings_vscode/wordpress_findings_codebase/run_wordpress_findings.py")
        self.assertIn('"smoke"', text)
        self.assertIn("tests/smoke_test.py", text)
        self.assertIn("recommended-order", text)


if __name__ == "__main__":
    unittest.main()
