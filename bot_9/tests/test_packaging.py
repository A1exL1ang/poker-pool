import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('package_bot', ROOT / 'tools/package_bot.py')
package_bot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package_bot)


class PackagingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix='bot9-package-test-', dir=ROOT)
        cls.archive = Path(cls.directory.name) / 'bot.zip'
        cls.report = package_bot.build_package(cls.archive)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_isolated_network_disabled_no_site_dependencies(self):
        smoke = self.report['isolated_smoke']
        self.assertEqual(smoke['smoke_decisions'], 16)
        self.assertEqual(smoke['agent_errors'], 0)
        self.assertEqual(smoke['network_attempts'], 0)
        self.assertEqual(smoke['third_party_modules'], 0)
        self.assertTrue(smoke['model_loaded'])
        self.assertEqual(smoke['jsonl_lines'],2)

    def test_content_hashes_and_runtime_whitelist(self):
        self.assertLess(self.report['archive_bytes'], 100_000_000)
        self.assertLess(self.report['uncompressed_bytes'], 100_000_000)
        with zipfile.ZipFile(self.archive) as archive:
            manifest = json.loads(archive.read('MANIFEST.json'))
            expected = set(package_bot.runtime_files()) | {'MANIFEST.json'}
            self.assertEqual(set(archive.namelist()), expected)
            for name, record in manifest['files'].items():
                data = archive.read(name)
                self.assertEqual(hashlib.sha256(data).hexdigest(), record['sha256'])
                self.assertEqual(len(data), record['bytes'])

    def test_import_graph_rejects_external_or_dynamic_imports(self):
        with tempfile.TemporaryDirectory(prefix='bot9-import-test-', dir=ROOT) as tmp:
            root = Path(tmp)
            for name in package_bot.SOURCE_FILES:
                (root / name).write_text('import math\n', encoding='utf-8')
            self.assertEqual(package_bot.check_import_graph(root)['bot.py'], ['math'])
            for bad in ('import numpy\n', '__import__("socket")\n', 'importlib.import_module("x")\n'):
                (root / 'bot.py').write_text(bad, encoding='utf-8')
                with self.assertRaises(ValueError):
                    package_bot.check_import_graph(root)


if __name__ == '__main__':
    unittest.main()
