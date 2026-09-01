#!/usr/bin/env python3
"""Adaptive Bench2Drive closed-loop evaluation of one planner with SC-IRT (UP).

Instead of rolling out all 220 routes, the driver asks `scirt.live` for the
next route, runs it through the Bench2Drive leaderboard evaluator (one
route per invocation, CARLA spawned by the evaluator itself, crash retries
as in scripts/sh/run_pdmlite_rollout.sh), reads the outcome from the
checkpoint JSON, updates the posterior and stops when the calibrated risk
c * R1 <= eps (or the route budget is exhausted). Everything is logged to
OUT/adaptive_log.json and the run resumes from it.

    python tools/b2d_adaptive_eval.py --name my_planner --out /data1/jeongtae/b2d_adaptive/my_planner \
        --agent $LEADERBOARD_ROOT/team_code/my_agent.py --agent-config /path/to/config \
        --gpu 2 --gpu-rank 2 --eps 0.03 --max-routes 110

    python tools/b2d_adaptive_eval.py --dry-run PLANNER_NAME --eps 0.03   # simulate from the matrix

Pass rule = the response-matrix rule: status in {Completed, Perfect} and
every infraction list empty except min_speed_infractions.
"""
import argparse
import copy
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scirt.live import LiveEvaluator

DEFAULTS = dict(
    carla_root='/data1/jeongtae/carla915',
    work_dir='/home/jeongtae/IRT/repos/carla_garage_b2d/Bench2Drive',
    python='/home/jeongtae/miniconda3/envs/b2d_roll/bin/python',
    xml='/home/jeongtae/IRT/repos/carla_garage_b2d/Bench2Drive/leaderboard/data/bench2drive220.xml',
)


def is_success(rec):
    if rec.get('status') not in ('Completed', 'Perfect'):
        return False
    inf = rec.get('infractions', {})
    return all((len(v) if hasattr(v, '__len__') else v) == 0 for k, v in inf.items() if k != 'min_speed_infractions')


def write_route_xml(master, route_ids, dst):
    root = ET.parse(master).getroot()
    out = ET.Element(root.tag, root.attrib)
    by_id = {r.get('id'): r for r in root.iter('route')}
    for rid in route_ids:
        out.append(copy.deepcopy(by_id[str(rid)]))
    ET.ElementTree(out).write(dst, encoding='utf-8', xml_declaration=True)


def run_evaluator(a, xml, ckpt, log):
    lb = f'{a.work_dir}/leaderboard'
    env = dict(os.environ)
    env.update({
        'CARLA_ROOT': a.carla_root, 'WORK_DIR': a.work_dir,
        'SCENARIO_RUNNER_ROOT': f'{a.work_dir}/scenario_runner', 'LEADERBOARD_ROOT': lb,
        'PYTHONPATH': f'{a.carla_root}/PythonAPI/carla:{a.work_dir}/scenario_runner:{lb}:' + env.get('PYTHONPATH', ''),
        'IS_BENCH2DRIVE': 'True', 'CUDA_VISIBLE_DEVICES': str(a.gpu),
    })
    for kv in a.env:
        k, v = kv.split('=', 1)
        env[k] = v
    cmd = [a.python, f'{lb}/leaderboard/leaderboard_evaluator.py',
           f'--routes={xml}', '--repetitions=1', f'--track={a.track}', f'--checkpoint={ckpt}',
           f'--agent={a.agent}', f'--agent-config={a.agent_config}', '--debug=0', '--resume=True',
           f'--port={a.port}', f'--traffic-manager-port={a.tm_port}', f'--gpu-rank={a.gpu_rank}',
           f'--timeout={a.timeout}']
    with open(log, 'a') as lf:
        lf.write(f'\n=== {time.strftime("%F %T")} {" ".join(cmd)}\n')
        lf.flush()
        rc = subprocess.call(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    subprocess.call(['pkill', '-9', '-f', f'carla-rpc-port={a.port}'])
    time.sleep(5)
    return rc


def outcomes_from_checkpoint(ckpt):
    if not Path(ckpt).exists():
        return {}
    recs = json.load(open(ckpt))['_checkpoint']['records']
    out = {}
    for rec in recs:
        rid = rec['route_id'].replace('RouteScenario_', '').split('_rep')[0]
        out[rid] = (is_success(rec), rec.get('status'), rec.get('scores', {}).get('score_composed'))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=False, default=None, help='planner tag (excluded from the bank if it is in the matrix)')
    ap.add_argument('--out', default=None, help='output dir (default: <repo>/results/live_<name>)')
    ap.add_argument('--agent', default='')
    ap.add_argument('--agent-config', default='')
    ap.add_argument('--track', default='SENSORS')
    ap.add_argument('--gpu', type=int, default=2, help='CUDA index for the agent')
    ap.add_argument('--gpu-rank', type=int, default=2, help='Vulkan adapter index for CARLA (see run_pdmlite_rollout.sh)')
    ap.add_argument('--port', type=int, default=21000)
    ap.add_argument('--tm-port', type=int, default=41000)
    ap.add_argument('--timeout', type=float, default=600.0)
    ap.add_argument('--env', action='append', default=[], help='extra KEY=VAL for the evaluator process')
    ap.add_argument('--eps', type=float, default=0.03, help='stop when c * R1 <= eps')
    ap.add_argument('--max-routes', type=int, default=110)
    ap.add_argument('--batch', type=int, default=1, help='routes per evaluator invocation (no re-planning inside a batch)')
    ap.add_argument('--max-tries', type=int, default=5, help='evaluator relaunches per batch on crash')
    ap.add_argument('--risk-cache', default=None)
    ap.add_argument('--no-risk-scale', action='store_true', help='use raw R1 (c = 1) instead of the LOO-calibrated scale')
    ap.add_argument('--dry-run', default=None, metavar='PLANNER', help='simulate outcomes from the response matrix for this planner')
    for k, v in DEFAULTS.items():
        ap.add_argument(f'--{k.replace("_", "-")}', default=v)
    a = ap.parse_args()

    name = a.dry_run or a.name
    assert name, '--name or --dry-run required'
    out = Path(a.out or Path(__file__).resolve().parents[1] / 'results' / f'live_{name}')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'xml').mkdir(exist_ok=True)
    (out / 'checkpoints').mkdir(exist_ok=True)
    ev = LiveEvaluator(exclude=(name,))
    if a.no_risk_scale:
        ev.c = 1.0
        print('[warn] --no-risk-scale: stopping on raw R1 (the mean error, not a tail bound)')
    else:
        ev.calibrate_risk(a.risk_cache)
    truth = None
    if a.dry_run:
        k = ev.panel.names.index(a.dry_run)
        truth = {r: ev.panel.Y[(r, k)] for r in ev.routes if (r, k) in ev.panel.Y}
        print(f'[dry-run] {a.dry_run}: true SR {np.mean(list(truth.values())):.4f} on {len(truth)} routes')

    logp = out / 'adaptive_log.json'
    log = json.load(open(logp)) if logp.exists() else []
    for e in log:                                        # resume
        ev.observe(e['route_id'], e['passed'])
    if log:
        print(f'[resume] {len(log)} routes replayed; {ev.estimate()}')

    t = len(log)
    while t < a.max_routes and not ev.done(a.eps):
        rids = ev.next(k=a.batch, allowed=set(truth) if truth is not None else None)
        if not rids:
            break
        if truth is not None:
            res = {r: (bool(truth[r]), 'dry-run', None) for r in rids}
        else:
            step = len(log)
            xml, ckpt = out / 'xml' / f'step_{step:03d}.xml', out / 'checkpoints' / f'step_{step:03d}.json'
            write_route_xml(a.xml, rids, xml)
            res = {}
            for attempt in range(a.max_tries):
                run_evaluator(a, xml, ckpt, out / 'evaluator.log')
                res = outcomes_from_checkpoint(ckpt)
                if all(r in res and not str(res[r][1]).startswith('Failed - Simulation crashed') for r in rids):
                    break
                print(f'[step {step}] evaluator did not finish {rids} (attempt {attempt + 1}/{a.max_tries}); relaunching', flush=True)
            bad = [r for r in rids if r not in res or str(res[r][1]).startswith('Failed - Simulation crashed')]
            if bad:
                sys.exit(f'[step {step}] {bad} did not finish after {a.max_tries} evaluator launches (see {ckpt}); '
                         f'nothing observed for them — fix the simulator and rerun to resume')
        for r in rids:
            passed, status, score = res[r]
            ev.observe(r, passed)
            est = ev.estimate()
            log.append({'t': len(log) + 1, 'route_id': r, 'type': ev.panel.sn[r], 'passed': bool(passed),
                        'status': status, 'score': score, **est})
            json.dump(log, open(logp, 'w'), indent=1)
            print(f'[t={len(log):3d}] route {r:>6s} {ev.panel.sn[r]:28s} {"PASS" if passed else "fail"}  '
                  f'SR_hat {est["sr_hat"]:.3f}  R1 {est["r1"]:.4f}  risk {est["risk"]:.4f}  types {est["types_covered"]}', flush=True)
            t = len(log)
    est = ev.estimate()
    summ = {'planner': name, 'eps': a.eps, 'stopped_by': 'risk' if ev.done(a.eps) else 'budget', **est,
            'routes': [e['route_id'] for e in log]}
    if truth is not None:
        summ['true_sr'] = float(np.mean(list(truth.values())))          # over the planner's recorded routes
        summ['truth_routes'] = len(truth)
        summ['abs_err'] = abs(summ['true_sr'] - est['sr_hat'])
    json.dump(summ, open(out / 'summary.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != 'routes'}, indent=1))


if __name__ == '__main__':
    main()
