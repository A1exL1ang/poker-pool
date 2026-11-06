"""Import unchanged submissions without sharing their unqualified module names.

Each submission gets a private package. Its source modules receive an import
function which redirects *only* names present in that submission's root. This
also covers delayed imports without modifying process-global builtins/sys.path.
__file__ remains the real source path, so frozen relative assets still load.
"""
import builtins
import importlib
import importlib.abc
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_IMPORT = builtins.__import__
_ROOTS = {}
_LOCAL_NAMES = {}


class _Source(importlib.abc.Loader):
    def __init__(self, path, namespace, root):
        self.path, self.namespace, self.root = path, namespace, root

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        def local_import(name, globals=None, locals=None, fromlist=(), level=0):
            if not level:
                top = name.partition('.')[0]
                if top in _LOCAL_NAMES[self.namespace]:
                    qualified = self.namespace + '.' + name
                    imported = _IMPORT(qualified, globals, locals, fromlist or ('*',), 0)
                    return imported if fromlist else sys.modules[self.namespace + '.' + top]
            return _IMPORT(name, globals, locals, fromlist, level)

        module.__builtins__ = dict(vars(builtins), __import__=local_import)
        # compile(bytes) honors the source encoding declaration; no source edits.
        exec(compile(self.path.read_bytes(), str(self.path), 'exec'), module.__dict__)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        namespace, _, tail = fullname.partition('.')
        root = _ROOTS.get(namespace)
        if root is None:
            return None
        directory = root.joinpath(*tail.split('.')) if tail else root
        if directory.is_dir():
            source = directory / '__init__.py'
            if source.is_file():
                return importlib.util.spec_from_file_location(
                    fullname, source, loader=_Source(source, namespace, root),
                    submodule_search_locations=[str(directory)])
            spec = importlib.machinery.ModuleSpec(fullname, loader=None, is_package=True)
            spec.submodule_search_locations = [str(directory)]
            return spec
        source = directory.with_suffix('.py')
        if source.is_file():
            return importlib.util.spec_from_file_location(
                fullname, source, loader=_Source(source, namespace, root))
        return None


_FINDER = _Finder()
sys.meta_path.insert(0, _FINDER)


def load_bot_module(number, module='agent'):
    namespace = 'isolated_submission_' + str(int(number))
    root = _ROOTS.setdefault(namespace, ROOT / ('bot_' + str(int(number))))
    if namespace not in _LOCAL_NAMES:
        _LOCAL_NAMES[namespace] = {p.stem if p.is_file() else p.name for p in root.iterdir() if p.is_dir() or p.suffix == '.py'}
    return importlib.import_module(namespace + '.' + module)
