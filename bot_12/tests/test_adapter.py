import json
from pathlib import Path
import subprocess
import sys
import unittest


class AdapterTests(unittest.TestCase):
    def test_bad_json_does_not_prevent_later_reset(self):
        root=Path(__file__).resolve().parents[1]
        p=subprocess.run([sys.executable,str(root/'run_bot.py')],
                         input='null\n[]\n{broken\n{"command":"reset","seed":120}\n',
                         text=True,capture_output=True,check=True,cwd=root)
        self.assertEqual([json.loads(s) for s in p.stdout.splitlines()],
                         [{'action':'fold'}]*3+[{'ok':True}])


if __name__=='__main__':
    unittest.main()
