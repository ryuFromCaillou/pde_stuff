"""Check notebook path setup without executing training or rewriting artifacts."""
import ast
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebook/diagnostics/burgers_minimal_discovery_story.ipynb'


class NotebookPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = [''.join(cell['source']) for cell in
                       json.loads(NOTEBOOK.read_text())['cells']
                       if cell['cell_type'] == 'code']
        cls.setups = []
        for source in cls.sources:
            nodes = ast.parse(source).body
            start = next((i for i, node in enumerate(nodes)
                          if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'repo_root'
                                  for t in node.targets)), None)
            if start is not None:
                # Root assignment, missing-root guard, and import-path setup.
                cls.setups.append(compile(ast.Module(body=nodes[start:start + 3],
                                                     type_ignores=[]), '<root setup>', 'exec'))

    def test_root_setup_from_root_and_notebook_directory(self):
        self.assertEqual(len(self.setups), 4)
        for cwd in (ROOT, NOTEBOOK.parent):
            for code in self.setups:
                with self.subTest(cwd=cwd), patch.object(Path, 'cwd', return_value=cwd), patch.object(sys, 'path', []):
                    before = Path(os.getcwd())
                    env = {'Path': Path, 'sys': sys}
                    exec(code, env)
                    self.assertEqual(env['repo_root'], ROOT)
                    self.assertEqual(sys.path[0], str(ROOT))
                    self.assertEqual(Path(os.getcwd()), before)

    def test_outside_repository_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            for code in self.setups:
                with patch.object(Path, 'cwd', return_value=Path(directory)):
                    with self.assertRaisesRegex(RuntimeError, 'repository root'):
                        exec(code, {'Path': Path, 'sys': sys})

    def test_repository_paths_are_absolute_and_exist(self):
        checked = 0
        for source in self.sources:
            tree = ast.parse(source)
            self.assertNotIn('chdir(', source)
            self.assertNotIn('../../run_results', source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'Path':
                    self.fail('Notebook artifact paths must use repo_root, not cwd-relative Path(...)')
                if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
                    and isinstance(node.left, ast.Name)
                    and node.left.id in {'repo_root', 'p19bii_root', 'p22_root', 'phase24_root'}
                    and isinstance(node.right, ast.Constant) and isinstance(node.right.value, str)):
                    path = ROOT / node.right.value
                    self.assertTrue(path.is_absolute())
                    self.assertTrue(path.exists(), str(path))
                    checked += 1
        self.assertGreater(checked, 20)


if __name__ == '__main__':
    unittest.main()
