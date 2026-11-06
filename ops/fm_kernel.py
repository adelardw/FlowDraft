"""Private Kaggle session body. Credentials are never embedded in this file."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

root = Path('/kaggle/working/FlowDraft')
inputs = Path('/kaggle/input')
import torch
if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (7, 0):
    raise RuntimeError('compatible GPU required; no training attempted')
print('FM_GPU', torch.cuda.get_device_name(0), flush=True)
root.mkdir(exist_ok=True)
archives = list(inputs.rglob('flowdraft-fm-code.tar.gz'))
if archives:
    with tarfile.open(archives[0]) as tar:
        tar.extractall(root, filter='data')
else:
    modules = list(inputs.rglob('flowdraft_fm_bank.py'))
    if len(modules) != 1:
        raise RuntimeError('cannot locate the mounted fixed-bank code tree')
    tree = modules[0].parents[2]
    for name in ('src','ops'):
        shutil.copytree(tree/name, root/name, dirs_exist_ok=True)
    shutil.copy2(tree/'pyproject.toml', root/'pyproject.toml')
for name, content in globals().get('SOURCE_UPDATES', {}).items():
    if name != 'ops/fm_entry.py':
        raise ValueError('unexpected session entry override')
    (root/name).write_text(content)
    print('FM_ENTRY_SHA256', hashlib.sha256(content.encode()).hexdigest(), flush=True)
os.chdir(root)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'lightning==2.6.5',
                'transformers==5.12.1', 'hydra-core', 'loguru', 'tensorboard', 'accelerate'], check=True)
token_file = next(inputs.rglob('tokens.pt'))
manifest = json.loads((token_file.parent/'manifest.json').read_text())
if hashlib.sha256(token_file.read_bytes()).hexdigest() != manifest['tokens_sha256']:
    raise RuntimeError('token pool checksum mismatch')
out = Path('/kaggle/working/fm-output')
out.mkdir(exist_ok=True)
resume_files = list(inputs.rglob('resume.ckpt'))
resume_args = []
expected = int(globals().get('EXPECTED_STEP', 0))
if expected:
    if len(resume_files) != 1:
        raise RuntimeError('exactly one resume checkpoint required')
    checkpoint = torch.load(resume_files[0], map_location='cpu', weights_only=False)
    if checkpoint['global_step'] != expected or checkpoint['hyper_parameters']['trainer']['max_steps'] != 60000:
        raise RuntimeError('wrong checkpoint version or LR horizon mounted')
    bank = resume_files[0].parent/'endpoints.sqlite'
    if not bank.exists():
        raise RuntimeError('fixed endpoint bank missing from resume dataset')
    shutil.copy2(bank, out/'endpoints.sqlite')
    resume_args = [f'resume_from_checkpoint={resume_files[0]}']
elif resume_files:
    raise RuntimeError('unexpected resume data at step zero')
env = os.environ.copy()
env['PYTHONPATH'] = str(root)
env['FLOWDRAFT_DF_ATTENTION'] = 'dense'
env['TOKENIZERS_PARALLELISM'] = 'false'
env['FD_FM_SESSION_HOURS'] = str(globals().get('SESSION_HOURS', 3.0))
command = [sys.executable, '-u', 'ops/fm_entry.py', '--config-path', str(root/'src/configs'),
           '+experiment=qwen06_flowdraft_fm_bank',
           f'data.fm_tokens_path={token_file}', f'output_dir={out}',
           '+trainer.enable_progress_bar=false', '+trainer.num_sanity_val_steps=0',
           '+trainer.check_val_every_n_epoch=null', *resume_args]
print('FM_LAUNCH', json.dumps({'expected_step':expected,'max_steps':60000,'anchors':4,
                              'examples':80000,'bank_size':16,'command':command}), flush=True)
completed = subprocess.run(command, env=env)
if completed.returncode:
    raise RuntimeError(f'training failed with exit code {completed.returncode}')
summary = json.loads((out/'session-summary.json').read_text())
if summary['step_after'] <= expected:
    raise RuntimeError('session made no optimizer progress')
for p in out.glob('*.ckpt'):
    if p.name != 'last.ckpt':
        p.unlink()
print('FM_KERNEL_COMPLETE', json.dumps(summary), flush=True)
