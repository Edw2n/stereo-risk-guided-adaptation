import json, time, sys
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import numpy as np, h5py
from PIL import Image
from sensor_v4_common import ROOT, read_config, write_json, digest, seed, storage
from sensor_v4_images import corrupt, variant, candidate_bank, cached_rgb, image_stats
from sensor_v4_metrics import native_maps, sufficient
from sensor_v4_signals import RiskExtractor
from table2_pool_v4 import get_shared_coordinates, write_pool, save_common_support, load_samples

def prepare(tasks, config, cache):
    begin = time.perf_counter()
    views, hashes = [[], []], []
    for task in tasks:
        r, condition = task['record'], task['condition']
        row_hash = {}
        for eye, kind in enumerate(('frame_left', 'frame_right')):
            path = ROOT/r['assets'][kind]['path']
            # Full Phase A may overlap the authorized atomic asset downloader.
            if task.get('wait_for_assets'):
                deadline=time.monotonic()+7200
                while not all((ROOT/a['path']).exists() for a in r['assets'].values()):
                    if time.monotonic()>deadline:raise TimeoutError('Waiting for authorized manifest assets exceeded 2h')
                    time.sleep(2)
            # Source ZIP CRC and immutable protocol hash identify cached RGB.
            key = f'{r["scene"]}/{r["frame"]}_{condition}_{eye}'
            rgb = cached_rgb(cache/'incoming'/f'{key}.npy', lambda: corrupt(np.asarray(Image.open(path).convert('RGB')), condition, r, eye, config))
            candidate = task.get('candidate')
            if candidate:
                incoming = rgb
                rgb = cached_rgb(cache/'variants'/f'{key}_{candidate["id"]}.npy', lambda: variant(incoming, candidate, r, condition, eye, config))
            if rgb.shape != (config['inference']['height'], config['inference']['width'], 3):
                raise ValueError('Input is not native resolution')
            views[eye].append(rgb)
            row_hash[kind] = digest(rgb.tobytes())
        hashes.append(row_hash)
    return tasks, views, hashes, time.perf_counter()-begin


def evaluate(payload):
    tasks, views, hashes, loading = payload['prepared']
    config, name = payload['config'], payload['model']
    extractor = RiskExtractor(name, config)
    rows = []
    for i, task in enumerate(tasks):
        start = time.perf_counter()
        rec = task['record']
        with h5py.File(ROOT/rec['assets']['disp1_left']['path'], 'r') as f:
            gt = np.asarray(f['disparity'])
        pred = payload['prediction'][i]
        error, d1, valid, common = native_maps(pred, gt)
        native = {k: v[i] for k, v in payload['signals'].items()}
        maps = extractor.maps(native, pred, views[0][i], rec)
        means = extractor.observation_scores(maps)
        row = {'pair_id': rec['pair_id'], 'scene': rec['scene'], 'frame': rec['frame'], 'split': rec['split'],
               'model': name, 'checkpoint_sha256': config['models'][name]['checkpoint_sha256'],
               'table': task['table'], 'condition': task['condition'], 'severity': 'published_single_setting',
               'candidate_id': task.get('candidate', {}).get('id', 'incoming'),
               'full': sufficient(error, d1, valid), 'common_support': sufficient(error, d1, common),
               'risk_means': means, 'input_sha256': hashes[i], 'timing_per_sample': {k: v/len(tasks) for k, v in payload['timing'].items()},
               'preprocessing_batch_seconds': loading, 'batch_size': len(tasks)}
        if task['table'] == 'T12':
            domain = valid.copy(); domain[:,:4] = False
            for key, risk in maps.items():
                if not np.isfinite(risk[domain]).all():
                    raise ValueError(f'Unexpected nonfinite risk map {key}; do not change the evaluation mask')
            pool_root = Path(payload['sample_dir']).parent/'analysis_pool'
            ids = get_shared_coordinates(pool_root, rec, valid, config)
            save_common_support(pool_root,rec,ids,common)
            row['analysis_pool'] = write_pool(pool_root, rec, name, task['condition'], ids,
                error.ravel()[ids], d1.ravel()[ids], {k: v.ravel()[ids] for k, v in maps.items()}, config)
            row['sampled_pixels'] = min(len(ids), config['metrics']['table2_sample_pixels_per_pair'])
            row['cached_pool_pixels'] = len(ids)
        elif task['table'] == 'T3':
            # Rank voting and fusion require the actual H x W maps. Scalar means
            # remain descriptive metadata only and cannot recover these methods.
            dense_path = Path(payload['sample_dir'])/f'{rec["scene"]}_{rec["frame"]}_{task["condition"]}_{row["candidate_id"]}_dense.npz'
            with dense_path.open('xb') as f:
                np.savez(f, disparity=pred, **{k: maps[k] for k in extractor.internal})
            row['dense_path'] = str(dense_path)
        row['metric_and_writing_seconds'] = time.perf_counter()-start
        rows.append(row)
    return rows


def model_worker(name, config, records, phase, output, k=None, selector_only=False):
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(20260925)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    mcfg = config['models'][name]
    checkpoint = ROOT/mcfg['checkpoint']
    if digest(checkpoint.read_bytes()) != mcfg['checkpoint_sha256']:
        raise ValueError('Checkpoint hash mismatch')
    batch = mcfg['batch']
    free, total = torch.cuda.mem_get_info()
    reserved = 62.684 if name == 'acv' and batch == 4 else 12.522
    if free/2**30 < reserved+config['inference']['minimum_free_margin_gib']:
        raise RuntimeError(f'Insufficient free VRAM for validated batch {batch}: {free/2**30:.2f} GiB; no unsafe launch')
    start = time.perf_counter()
    if name == 'raft':
        from raft_adapter import RAFTAdapter
        model = RAFTAdapter(checkpoint, mcfg['iterations'], last_n=mcfg['last_n'])
    else:
        from acv_adapter import ACVAdapter
        model = ACVAdapter(checkpoint, mcfg['maxdisp'])
    load_seconds = time.perf_counter()-start
    conditions = config['preflight']['conditions'] if phase == 'preflight' else config['conditions']
    ordered=sorted(records,key=lambda r:(r['split']!='dev',not all((ROOT/a['path']).exists() for a in r['assets'].values()),r['pair_id']))
    tasks = [{'record': r, 'condition': c, 'table': 'T12','wait_for_assets':phase=='A'}
             for split in ('dev','test') for c in conditions for r in ordered if r['split']==split]
    if selector_only:
        tasks = []
    if phase == 'preflight':
        bank = candidate_bank(config, ids=config['preflight']['variant_candidate_ids']) if k is None else candidate_bank(config, k=k)
        tasks += [{'record': r, 'condition': c, 'table': 'T3', 'candidate': v}
                  for r in records[:config['preflight']['variant_pairs']]
                  for c in ('clean', 'gaussian_noise') for v in bank]
    cache_id = digest({'config': config, 'generator': digest((ROOT/'scripts/sensor_v4_images.py').read_bytes()),
                       'asset_inventory': [(r['pair_id'], [a['crc32'] for a in r['assets'].values()]) for r in records]})[:16]
    cache = storage(config, 'cache', 'sensor_v4', cache_id)
    sample_dir = storage(config, 'intermediates', output.name, name)
    model_output = output/name; model_output.mkdir()
    sealed=storage(config,'intermediates',output.name,'sealed_test')/f'{name}.jsonl'
    chunks = [tasks[i:i+batch] for i in range(0, len(tasks), batch)]
    meta = {'model': name, 'config_sha256': digest(config), 'record_ids': sorted({t['record']['pair_id'] for t in tasks}),
            'sealed_test_records':str(sealed),'report_policy':'dev only until manual freeze',
            'selector_only': selector_only,
            'phase': phase, 'batch': batch, 'load_seconds': load_seconds, 'cache_path': str(cache),
            'torch': torch.__version__, 'numpy': np.__version__, 'GPU': torch.cuda.get_device_name(),
            'free_vram_before_load_gib': free/2**30, 'started_utc': datetime.now(timezone.utc).isoformat(),
            'command': sys.argv, 'script_hashes': {p.name: digest(p.read_bytes()) for p in (ROOT/'scripts').glob('*v4*.py')},
            'metric_reference': 'Spring public server 73a3a6bc15d4eb82735c7673630004c29f96cf8b/get_errors_stereo'}
    from PIL import features, __version__ as pillow_version
    import cv2
    meta['image_backends']={'Pillow':pillow_version,'libjpeg':features.version_codec('jpg'),'OpenCV':cv2.__version__}
    write_json(model_output/'run.json', meta)
    torch.cuda.reset_peak_memory_stats()
    clock = time.perf_counter(); completed = 0
    with ThreadPoolExecutor(max_workers=2) as prep, ThreadPoolExecutor(max_workers=2) as cpu:
        queue, pending = deque(), deque()
        it = iter(chunks)
        for chunk in list(chunks[:2]):
            queue.append(prep.submit(prepare, next(it), config, cache))
        with (model_output/'samples.jsonl').open('x') as result, sealed.open('x') as test_result:
            def collect(future):
                nonlocal completed
                rows = future.result()
                if completed==0 and phase=='A':
                    proof=[]
                    for row in rows:
                        saved=load_samples(row);analysis=load_samples(row,config['metrics']['table2_sample_pixels_per_pair'])
                        if len(saved['error'])!=50000 or len(analysis['error'])!=20000:raise AssertionError('Initial real cache gate failed')
                        if not np.array_equal(saved['pixel_index'][:20000],analysis['pixel_index']):raise AssertionError('Analysis is not nested prefix')
                        if 'common_support' not in saved:raise AssertionError('Secondary mask missing')
                        proof.append({'pair_id':row['pair_id'],'model':name,'pool_pixels':len(saved['error']),
                            'analysis_pixels':len(analysis['error']),'nested_prefix':True,'secondary_mask_saved':True})
                    write_json(model_output/'FIRST_BATCH_CACHE_PASS.json',proof)
                    print('FIRST_BATCH_CACHE_PASS',name,'saved 50000, analysis 20000',flush=True)
                for row in rows:
                    target=test_result if row['split']=='test' else result
                    target.write(json.dumps(row, allow_nan=False)+'\n')
                result.flush();test_result.flush(); completed += len(rows)
                print(json.dumps({'model': name, 'completed': completed, 'total': len(tasks), 'seconds': time.perf_counter()-clock}), flush=True)
            while queue:
                prepared = queue.popleft().result()
                following = next(it, None)
                if following:
                    queue.append(prep.submit(prepare, following, config, cache))
                prediction, signals, timing = model.predict_batch(*prepared[1])
                pending.append(cpu.submit(evaluate, {'prepared': prepared, 'config': config, 'model': name,
                    'prediction': prediction, 'signals': signals, 'timing': timing, 'sample_dir': str(sample_dir)}))
                if completed==0 or len(pending) >= 2:
                    collect(pending.popleft())
            while pending:
                collect(pending.popleft())
    meta.update({'wall_seconds': time.perf_counter()-clock, 'evaluations': len(tasks),
                 'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                 'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30, 'complete': True})
    write_json(model_output/'run.json', meta)
    (model_output/'COMPLETE').write_text('ok\n')

