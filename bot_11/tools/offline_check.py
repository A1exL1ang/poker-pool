"""Extract the offline archive and test it with isolated, site-free Python."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import zipfile


def child(root):
    sys.path.insert(0,str(root))
    import socket
    from unittest.mock import patch
    manifest = json.loads((root/'benchmarks/runtime_manifest.json').read_text())
    for name, record in manifest.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==record['sha256'],name
    with patch.object(socket,'socket',side_effect=RuntimeError('offline socket access blocked')):
        from agent import Agent
        from tools.latency import capture, legal_action
        bot = Agent()
        with patch('builtins.open',side_effect=RuntimeError('decision file access blocked')), \
             patch('io.open',side_effect=RuntimeError('decision file access blocked')), \
             patch('os.open',side_effect=RuntimeError('decision file access blocked')):
            records, totals = capture(200,1319000,32,False)
            assert len(records)==200
            for _,obs in records:
                action = bot.act(obs)
                assert legal_action(obs,action),action
            assert bot.diagnostics['errors']==0
            assert totals['totals']['policy_error_fallbacks']==0
    import unittest
    log = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(root/'tests'))
    result = unittest.TextTestRunner(stream=log,verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError(log.getvalue())
    transport = subprocess.run([sys.executable,'-S',str(root/'agent.py')],
                               input=json.dumps(records[0][1])+'\n',text=True,
                               capture_output=True,check=True,timeout=30,cwd=root)
    assert legal_action(records[0][1],json.loads(transport.stdout))
    return {'python':sys.version,'isolated_mode':bool(sys.flags.isolated),
            'site_packages_disabled':bool(sys.flags.no_site),'runtime_manifest_verified':True,
            'network_blocked_during_startup_and_play':True,'file_io_blocked_during_play':True,
            'offline_decisions':len(records),'errors':bot.diagnostics['errors'],
            'tests_run':result.testsRun,'tests_skipped':len(result.skipped),
            'skipped_tests':[(str(test),reason) for test,reason in result.skipped],
            'json_lines_transport_verified':True,
            'tests_successful':result.wasSuccessful()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archive',default='bot_11_offline.zip')
    p.add_argument('--out',default='benchmarks/package_validation.json')
    p.add_argument('--child',action='store_true')
    p.add_argument('--root')
    args = p.parse_args()
    if args.child:
        print(json.dumps(child(Path(args.root).resolve())))
        return
    workspace = Path.cwd().resolve()
    archive = Path(args.archive).resolve()
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    target = (workspace/'.control'/f'offline_smoke_{digest[:12]}').resolve()
    if workspace not in target.parents:
        raise ValueError('extraction must remain within this workspace')
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        for name in names:
            rel = PurePosixPath(name)
            if rel.is_absolute() or '..' in rel.parts or '.control' in rel.parts:
                raise ValueError('unexpected archive path')
        assert z.testzip() is None
        z.extractall(target)
    run = subprocess.run([sys.executable,'-I','-S',str(target/'tools/offline_check.py'),
                          '--child','--root',str(target)],cwd=target,
                         capture_output=True,text=True,check=True,timeout=120)
    report = json.loads(run.stdout)
    report.update({'archive':str(archive.name),'sha256':digest,'bytes':archive.stat().st_size,
                   'archive_crc_verified':True,'archive_files':len(names)})
    Path(args.out).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
