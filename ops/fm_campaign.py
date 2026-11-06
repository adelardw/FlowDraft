"""Private fixed-bank campaign on the explicitly supplied Kaggle account.

The token stays in an ignored local file. Assets contain only source code and
the prepared token pool. Resume data includes the optimizer and endpoint bank.
"""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT/'kaggle_out/fm-bank60k'
PROFILE = ROOT/'kaggle_out/fm-account'
USER = 'wolfram74'
ASSETS = f'{USER}/flowdraft-fm-assets'
RESUME = f'{USER}/flowdraft-fm-state'


def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


def api():
    os.environ['KAGGLE_API_TOKEN'] = (PROFILE/'access_token').read_text().strip()
    for key in ('KAGGLE_USERNAME','KAGGLE_KEY','KAGGLE_CONFIG_DIR'):
        os.environ.pop(key, None)
    import requests.adapters
    original = requests.adapters.HTTPAdapter.send
    def bounded(self, request, **kwargs):
        if kwargs.get('timeout') is None:
            kwargs['timeout'] = (20,180)
        return original(self, request, **kwargs)
    requests.adapters.HTTPAdapter.send = bounded
    from kaggle.api.kaggle_api_extended import KaggleApi
    client = KaggleApi()
    client.authenticate()
    if client.config_values.get('username') != USER:
        raise RuntimeError('credential belongs to a different account')
    return client


def checked(result):
    error = result.get('error') if isinstance(result,dict) else getattr(result,'error',None)
    if error:
        raise RuntimeError(str(error))
    return result


def upload(client, folder, ref):
    (folder/'dataset-metadata.json').write_text(json.dumps({
        'id': ref, 'title': ref.split('/')[1], 'licenses':[{'name':'CC0-1.0'}]}))
    exists = any(str(x.ref) == ref for x in client.dataset_list(mine=True))
    if exists:
        checked(client.dataset_create_version(str(folder), version_notes='fixed-bank campaign', quiet=True, dir_mode='skip'))
    else:
        checked(client.dataset_create_new(str(folder), public=False, quiet=True, dir_mode='skip'))
    for _ in range(60):
        try:
            if 'ready' in str(client.dataset_status(ref)).lower():
                log(f'dataset ready: {ref}')
                return
        except Exception as error:
            response = getattr(error, 'response', None)
            if getattr(response, 'status_code', None) not in (403,404):
                raise
            # Newly created private datasets can take time to appear in status.
        time.sleep(10)
    raise RuntimeError(f'dataset not ready: {ref}')


def upload_assets(client):
    stage = WORK/'assets'
    stage.mkdir(parents=True, exist_ok=True)
    def keep(info):
        return None if '__pycache__' in info.name or info.name.endswith(('.pyc','.pyo')) else info
    with tarfile.open(stage/'flowdraft-fm-code.tar.gz','w:gz') as tar:
        for name in ('src','pyproject.toml','ops/fm_entry.py'):
            tar.add(ROOT/name, arcname=name, filter=keep)
    for name in ('tokens.pt','manifest.json'):
        shutil.copy2(ROOT/'kaggle_out/fm-prepared'/name, stage/name)
    code_sha = hashlib.sha256((stage/'flowdraft-fm-code.tar.gz').read_bytes()).hexdigest()
    (WORK/'asset-code-sha256').write_text(code_sha)
    upload(client, stage, ASSETS)
    log(f'assets uploaded; source sha256={code_sha}')


def save(state):
    temp = WORK/'state.tmp'
    temp.write_text(json.dumps(state,indent=2))
    temp.replace(WORK/'state.json')


def load():
    path = WORK/'state.json'
    return json.loads(path.read_text()) if path.exists() else {'step':0,'segment':1,'kernel':None,'phase':'ready'}


def submit(client, state, hours):
    slug = f"flowdraft-fm60k-a4-s{state['segment']:03d}"
    folder = WORK/'kernels'/slug
    folder.mkdir(parents=True,exist_ok=True)
    updates = {'ops/fm_entry.py': (ROOT/'ops/fm_entry.py').read_text()}
    (folder/'main.py').write_text(f"EXPECTED_STEP={state['step']}\nSESSION_HOURS={hours}\nSOURCE_UPDATES={updates!r}\n"+(ROOT/'ops/fm_kernel.py').read_text())
    ref = f'{USER}/{slug}'
    metadata = {'id':ref,'title':slug,'code_file':'main.py','language':'python','kernel_type':'script',
                'is_private':True,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4',
                'dataset_sources':[ASSETS]+([RESUME] if state['step'] else []),
                'competition_sources':[],'kernel_sources':[],'model_sources':[]}
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    result = checked(client.kernels_push(str(folder)))
    state.update(kernel=ref, phase='running', submitted_at=time.time())
    save(state)
    log(f'submitted {ref}; starting at {state["step"]}/60000; response={result}')


def collect(client, state):
    import torch
    folder = WORK/'segments'/f"{state['segment']:03d}"
    folder.mkdir(parents=True,exist_ok=True)
    client.kernels_output(state['kernel'], path=str(folder), force=True, quiet=True)
    summaries = list(folder.rglob('session-summary.json'))
    if len(summaries) != 1:
        raise RuntimeError('completed kernel has no unambiguous training summary')
    summary = json.loads(summaries[0].read_text())
    checkpoint_path = summaries[0].parent/'last.ckpt'
    bank_path = summaries[0].parent/'endpoints.sqlite'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    step = int(checkpoint['global_step'])
    if not state['step'] < step <= 60000 or step != summary['step_after'] or not bank_path.exists():
        raise RuntimeError('invalid or non-progressing session result')
    if checkpoint['hyper_parameters']['trainer']['max_steps'] != 60000:
        raise RuntimeError('incorrect LR horizon in checkpoint')
    upload_dir = WORK/'resume-upload'
    upload_dir.mkdir(exist_ok=True)
    shutil.copy2(checkpoint_path, upload_dir/'resume.ckpt')
    shutil.copy2(bank_path, upload_dir/'endpoints.sqlite')
    shutil.copy2(summaries[0], upload_dir/'session-summary.json')
    upload(client, upload_dir, RESUME)
    state.update(step=step,segment=state['segment']+1,kernel=None,phase='ready',last_summary=summary)
    save(state)
    log(f'persisted checkpoint + bank: {step}/60000')


def run(client, hours):
    lock = (WORK/'campaign.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = load()
    while state['step'] < 60000:
        if state['kernel'] is None:
            try:
                submit(client,state,hours)
            except RuntimeError as error:
                if 'quota' not in str(error).lower():
                    raise
                log(f'quota unavailable; retaining 60k schedule: {error}')
                deadline = time.time()+1800
                while time.time()<deadline:
                    time.sleep(min(30,deadline-time.time()))
                continue
        time.sleep(30)
        try:
            result = client.kernels_status(state['kernel'])
        except Exception as error:
            log(f'status temporarily unavailable: {type(error).__name__}')
            continue
        status = str(getattr(result,'status',result)).upper()
        if 'RUNNING' in status or 'QUEUED' in status:
            continue
        if 'COMPLETE' not in status:
            state['phase'] = 'failed'
            save(state)
            raise RuntimeError(f'kernel failed: {state["kernel"]}: {status}; automatic retries stopped')
        collect(client,state)
    state['phase'] = 'complete'
    save(state)
    log('training complete: 60000/60000')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['upload','run','status'])
    parser.add_argument('--hours',type=float,default=3.0)
    args=parser.parse_args()
    if not 0 < args.hours <= 9:
        raise ValueError('session must leave time for setup and saving')
    WORK.mkdir(parents=True,exist_ok=True)
    client=api()
    if args.action=='upload':
        upload_assets(client)
    elif args.action=='run':
        run(client,args.hours)
    else:
        state=load()
        print(json.dumps(state,indent=2))
        if state['kernel']:
            print(client.kernels_status(state['kernel']))


if __name__=='__main__':
    main()
