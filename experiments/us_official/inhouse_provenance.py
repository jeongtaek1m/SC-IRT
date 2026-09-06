#!/usr/bin/env python3
"""Write the provenance record for the four in-house Table 3A descriptor rows.

Rows documented: 'Route geometry' (16d), 'Agent density + kin.' (18d),
'Kinematics (cmdkin, 25d)' (25d) and 'Hand-crafted risk (cmdkin+gtrisk, 73d)'
(25 + 48).  None of the four has a published method behind it; this file records
what each column actually computes, which raw annotation fields it reads, and
whether the value comes from the PDM-Lite probe's own motion or from the
surrounding scene.  Nothing is recomputed except the per-column sanity
statistics, and no feature file is modified.

out: /data2/jeongtae/official_baselines/us_features/inhouse_provenance.json
"""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path('/home/jeongtae/SC-IRT')
OUT = Path('/data2/jeongtae/official_baselines/us_features/inhouse_provenance.json')

# source taxonomy ---------------------------------------------------------
SRC = {
    'probe_ego_motion': "PDM-Lite's own recorded ego pose / speed only",
    'probe_route_command': 'the route command + nav-target channel logged frame by frame along '
                           "the PDM-Lite rollout (route definition, but truncated / resampled by "
                           "where and how long the probe actually drove)",
    'probe_ego_x_command': "a ratio or weighting that mixes PDM-Lite's travelled distance with the "
                           'route command channel',
    'other_agents': 'other agents only (counts over all logged boxes; no ego reference)',
    'other_agents_at_probe_pose': "other agents evaluated in PDM-Lite's rollout ego frame "
                                  '(pose/heading), but not using the ego speed',
    'other_agents_at_probe_pose_and_speed': "other agents evaluated at PDM-Lite's rollout pose AND "
                                            'its realized speed',
    'other_agents_at_probe_pose_counterfactual_speed': "other agents evaluated at PDM-Lite's "
                                                       'rollout pose with the ego speed replaced by '
                                                       'a nominal 8 m/s',
}

EGO_FIELDS = "anno['x'], anno['y'], anno['theta'], anno['speed'] (10 Hz; PDM-Lite reference rollout)"
CMD_FIELDS = ("anno['command_near'], anno['command_far'], anno['x_command_near'], "
              "anno['y_command_near'], anno['x_command_far'], anno['y_command_far']")
BOX_FIELDS = ("anno['bounding_boxes'][*]: class, location, rotation, extent, speed, lane_id, road_id "
              "(ego_vehicle entry excluded / used as the ego reference)")

# ---- Route geometry, 16d ------------------------------------------------
ROUTEGEOM = [
    ('route_len', 'sum over valid frames of hypot(dx, dy) of the ego position', EGO_FIELDS, 'probe_ego_motion'),
    ('n_wp', 'number of distinct consecutive near-target waypoints (|d(x,y)| > 1e-3) seen along the rollout', CMD_FIELDS, 'probe_route_command'),
    ('sum_turn', 'sum of |wrap(d heading)| between consecutive polyline segments of the unique near-target waypoints, segments shorter than 0.5 m dropped', CMD_FIELDS, 'probe_route_command'),
    ('max_turn', 'max of the same per-vertex |wrap(d heading)|', CMD_FIELDS, 'probe_route_command'),
    ('turn_per_m', 'sum_turn / max(route_len, 1.0)', EGO_FIELDS + ' + ' + CMD_FIELDS, 'probe_ego_x_command'),
    ('max_turn_20m', 'largest cumulative |d heading| inside any 20 m stretch of the waypoint polyline', CMD_FIELDS, 'probe_route_command'),
    ('n_cmd_seg', 'number of runs of constant command_near along the rollout frames', CMD_FIELDS, 'probe_route_command'),
    ('seg_per_100m', 'n_cmd_seg / max(route_len / 100, 1e-6)', EGO_FIELDS + ' + ' + CMD_FIELDS, 'probe_ego_x_command'),
    ('n_uniq_cmd', 'number of distinct command_near values in 1..6 over the rollout', CMD_FIELDS, 'probe_route_command'),
    ('lookahead_mean', 'mean over frames of the distance between the far and the near command target', CMD_FIELDS, 'probe_route_command'),
] + [(f'distfrac_cmd{k}',
      f'ego travelled distance spent under command_near == {k}, divided by the total travelled distance '
      '(distance-weighted, so a stopped probe adds nothing)',
      EGO_FIELDS + ' + ' + CMD_FIELDS, 'probe_ego_x_command') for k in range(1, 7)]

# ---- Agent density + kin., 10 + 8d --------------------------------------
KIN = [
    ('v_mean', 'mean(speed)', EGO_FIELDS, 'probe_ego_motion'),
    ('v_max', 'max(speed)', EGO_FIELDS, 'probe_ego_motion'),
    ('absa_mean', 'mean(|diff(speed) / 0.1|)', EGO_FIELDS, 'probe_ego_motion'),
    ('absa_max', 'max(|diff(speed) / 0.1|)', EGO_FIELDS, 'probe_ego_motion'),
    ('absjerk_mean', 'mean(|diff(diff(speed) / 0.1) / 0.1|)', EGO_FIELDS, 'probe_ego_motion'),
    ('frac_stop', 'mean(speed < 0.5)', EGO_FIELDS, 'probe_ego_motion'),
    ('path_len', 'sum of the per-frame ego displacement norms', EGO_FIELDS, 'probe_ego_motion'),
    ('duration_s', 'number of annotated frames x 0.1 s, i.e. how long the PDM-Lite rollout lasted', EGO_FIELDS, 'probe_ego_motion'),
    ('dheading_mean', 'mean(|diff(unwrap(atan2(dy, dx)))|) with the heading taken from the ego position increments (not from theta)', EGO_FIELDS, 'probe_ego_motion'),
    ('dheading_max', 'max of the same per-frame heading increment', EGO_FIELDS, 'probe_ego_motion'),
]
DEN = [
    ('n_agents_mean', 'mean over frames of the number of non-ego bounding boxes (no class filter, no radius filter)', BOX_FIELDS, 'other_agents'),
    ('n_agents_max', 'max over frames of the same count', BOX_FIELDS, 'other_agents'),
    ('n_near10_mean', 'mean over frames of the number of non-ego boxes within 10 m of the ego position', BOX_FIELDS + ' + ' + EGO_FIELDS, 'other_agents_at_probe_pose'),
    ('n_near10_max', 'max over frames of the same count', BOX_FIELDS + ' + ' + EGO_FIELDS, 'other_agents_at_probe_pose'),
    ('n_walker_mean', "mean over frames of the number of boxes with class == 'walker'", BOX_FIELDS, 'other_agents'),
    ('frac_frames_with_walker', 'fraction of frames with at least one walker box', BOX_FIELDS, 'other_agents'),
    ('relspeed_mean', 'mean over frames of the per-frame mean of |agent speed - ego speed| over all non-ego boxes', BOX_FIELDS + ' + ' + EGO_FIELDS, 'other_agents_at_probe_pose_and_speed'),
    ('relspeed_max', 'max over frames of the per-frame max of |agent speed - ego speed|', BOX_FIELDS + ' + ' + EGO_FIELDS, 'other_agents_at_probe_pose_and_speed'),
]

# ---- cmdkin, 25d --------------------------------------------------------
CMDKIN = [
    ('v_mean', 'mean(v)', EGO_FIELDS, 'probe_ego_motion'),
    ('v_std', 'std(v), ddof = 0', EGO_FIELDS, 'probe_ego_motion'),
    ('v_max', 'max(v)', EGO_FIELDS, 'probe_ego_motion'),
    ('v_p10', 'percentile(v, 10), numpy linear interpolation', EGO_FIELDS, 'probe_ego_motion'),
    ('frac_stop', 'mean(v < 0.5)', EGO_FIELDS, 'probe_ego_motion'),
    ('a_std', 'std(a), a = diff(v) / 0.1', EGO_FIELDS, 'probe_ego_motion'),
    ('a_min', 'min(a)', EGO_FIELDS, 'probe_ego_motion'),
    ('a_max', 'max(a)', EGO_FIELDS, 'probe_ego_motion'),
    ('a_p05', 'percentile(a, 5)', EGO_FIELDS, 'probe_ego_motion'),
    ('absa_p95', 'percentile(|a|, 95)', EGO_FIELDS, 'probe_ego_motion'),
    ('yaw_mean', 'mean(|wrap(diff(theta - pi/2))|), radians PER FRAME (not divided by dt)', EGO_FIELDS, 'probe_ego_motion'),
    ('yaw_max', 'max(|wrap(diff(theta - pi/2))|)', EGO_FIELDS, 'probe_ego_motion'),
] + [(f'frac_cmd_near_{k}', f'fraction of rollout FRAMES with command_near == {k} (time-weighted: a stopped probe inflates whichever command it was dwelling in)', CMD_FIELDS, 'probe_route_command') for k in range(1, 7)] \
  + [(f'frac_cmd_far_{k}', f'fraction of rollout FRAMES with command_far == {k} (time-weighted)', CMD_FIELDS, 'probe_route_command') for k in range(1, 7)] \
  + [('cmd_switch_rate', 'mean(diff(command_near) != 0), i.e. command changes per frame', CMD_FIELDS, 'probe_route_command')]

# ---- gtrisk, 16 base channels x 3 aggregations = 48d --------------------
GT_BASE = [
    ('inv_ttc_vr', '1 / max(min_agent TTC, 0.5) with TTC the first root of the elliptical-footprint '
     'overlap under constant velocity, ego driving straight at its realized speed; footprint semi-axes '
     '(ego half-length + agent half-length + 0.3 m) x (ego half-width + agent half-width + 0.3 m); '
     'TTC = 0 when already overlapping', 'other_agents_at_probe_pose_and_speed'),
    ('rdec_vr', 'max over agents inside the ego corridor (|y_ego-frame| < B and x > 0) of dv^2 / (2 gap), '
     'dv = ego realized speed - agent longitudinal speed, gap = max(x - A, 0.5)', 'other_agents_at_probe_pose_and_speed'),
    ('n_conflict_vr', 'number of agents whose constant-velocity closest approach within a 4 s horizon '
     'lands inside the inflated ego footprint (normalized elliptical distance < 1), ego at its realized speed',
     'other_agents_at_probe_pose_and_speed'),
    ('inv_ttc_v0', 'same as inv_ttc_vr with the ego speed replaced by V0 = 8 m/s', 'other_agents_at_probe_pose_counterfactual_speed'),
    ('rdec_v0', 'same as rdec_vr with the ego speed replaced by V0 = 8 m/s', 'other_agents_at_probe_pose_counterfactual_speed'),
    ('n_conflict_v0', 'same as n_conflict_vr with the ego speed replaced by V0 = 8 m/s', 'other_agents_at_probe_pose_counterfactual_speed'),
    ('inv_gap', '1 / min over agents of max(range - ego silhouette radius - agent silhouette radius, 0.5)', 'other_agents_at_probe_pose'),
    ('closing_max', 'max over agents of the positive range rate towards a STATIONARY ego, '
     '-(px ux + py uy) / max(range, 0.1)', 'other_agents_at_probe_pose'),
    ('occl_frac', 'fraction of the +-60 deg forward field of view covered by the union of the agents angular extents', 'other_agents_at_probe_pose'),
    ('n_occluded', 'number of agents whose angular interval is hidden behind a nearer agent', 'other_agents_at_probe_pose'),
    ('n_near20', 'number of agents within 20 m of the ego', 'other_agents_at_probe_pose'),
    ('n_ped30', 'number of walkers within 30 m of the ego', 'other_agents_at_probe_pose'),
    ('n_cross40', 'number of agents within 40 m whose relative heading is between 45 and 135 deg', 'other_agents_at_probe_pose'),
    ('lane_share', 'number of agents ahead, within 40 m, with the same (lane_id, road_id) as the ego', 'other_agents_at_probe_pose'),
    ('agent_spd_std', 'std of agent speeds within 40 m (0 if fewer than 2)', 'other_agents_at_probe_pose'),
    ('n_agents60', 'number of vehicle/walker agents within the 60 m selection radius', 'other_agents_at_probe_pose'),
]
GT_STATS = [
    ('mean', 'plain mean over the sub-sampled time steps (every 5th frame, 2 Hz)'),
    ('p90', '90th percentile of a fixed-size bootstrap resample (m = 20 rows with replacement, 128 reps, seed 0), averaged over reps'),
    ('maxm', 'max of the same fixed-size bootstrap resample, averaged over reps'),
]
GTRISK = [(f'{sname}_{bname}',
           f'{sdesc} of the per-step channel: {bdesc}',
           BOX_FIELDS + ' + ' + EGO_FIELDS, bsrc)
          for sname, sdesc in GT_STATS for bname, bdesc, bsrc in GT_BASE]


def colstats(a):
    a = np.asarray(a, np.float64)
    return [{'mean': float(a[:, j].mean()), 'std': float(a[:, j].std()),
             'min': float(a[:, j].min()), 'max': float(a[:, j].max()),
             'n_zero': int((a[:, j] == 0).sum())} for j in range(a.shape[1])]


def block(cols, arr):
    st = colstats(arr)
    out = []
    for j, (name, formula, inputs, src) in enumerate(cols):
        out.append({'index': j, 'name': name, 'formula': formula, 'input': inputs,
                    'source': src, 'sanity': st[j]})
    return out


def tally(cols):
    t = {}
    for _, _, _, s in cols:
        t[s] = t.get(s, 0) + 1
    return t


def main():
    rg = np.load(REPO / 'data' / 'features' / 'eval_routegeom.npz', allow_pickle=True)
    kd = np.load(REPO / 'data' / 'b2d' / 'baseline_kin_den.npz', allow_pickle=True)
    ck = np.load(REPO / 'data' / 'features' / 'eval_cmdkin_stats.npz', allow_pickle=True)
    gt = np.load(REPO / 'data' / 'features' / 'eval_gtrisk.npz', allow_pickle=True)

    head = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'],
                          capture_output=True, text=True).stdout.strip()
    doc = {
        'title': 'Provenance of the four in-house descriptor rows of ATDrive Table 3A (US)',
        'written': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'generator': str(Path(__file__).resolve()),
        'atdrive_commit': head,
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'audit_verdict': (
            'No published method stands behind any of these four rows. They were designed inside '
            'this project. Every column is computed from the PDM-Lite reference rollout of the route '
            '(/data1/jeongtae/b2d_eval_sensors/route_<id>/anno/*.json.gz, 10 Hz), which is the same '
            'rollout the RelGraph encoder reads, so this is the protocol definition of a route '
            'representation and not leakage -- PDM-Lite is EXCLUDED_PLANNER and is not one of the 16 '
            'evaluated planners. But the rows must be described as PROBE-ROLLOUT descriptors, not as '
            'scene-only descriptors: the ego pose, the ego speed and the rollout length that enter '
            'them are PDM-Lite behaviour, and in three of the four sets a majority of the dimensions '
            'is either pure probe motion or an agent quantity measured in the probe frame.'),
        'route_bank': {'n_routes': 220, 'source_dir': '/data1/jeongtae/b2d_eval_sensors/route_<id>/anno',
                       'rate_hz': 10, 'reference_rollout': 'PDM-Lite (EXCLUDED_PLANNER)'},
        'how_scored': (
            'experiments/run_us.py:load_descriptor_arms -> two-stage Ridge plug-in: z-score on the '
            'calibration-type fold, Ridge(alpha = 100 if d > 10 else 10) fit b_hat ~ x, predict the '
            'evaluation types; pooled over 16 draws (640 route evaluations); AUROC / scene-MAE / '
            'Spearman rho.'),
        'sets': {},
    }

    doc['sets']['Route geometry'] = {
        'table_row': 'Route geometry',
        'file': str(REPO / 'data' / 'features' / 'eval_routegeom.npz'),
        'shape': list(rg['stats'].shape), 'dtype': str(rg['stats'].dtype),
        'names_format': 'route_<id>',
        'producer': '/home/jeongtae/SCIRT/b2d_irt/scripts/cmdkin_decompose.py:routegeom (lines 36-87), run(split="eval")',
        'producer_status': 'present in the repo, reads /data1/jeongtae/b2d_jepa/eval_feats/route_<id>.npz '
                           "{ego, cmd}, which b2d_irt/scripts/eval_pipeline.py:load_anno wrote from the raw anno",
        'published_method': None,
        'shipped_layout_string': str(rg['layout']),
        'verified': 'recomputed for all 220 routes by calling cmdkin_decompose.routegeom on the '
                    'eval_feats ego/cmd arrays: bit-for-bit identical to the shipped npz (0 routes differ).',
        'columns': block(ROUTEGEOM, rg['stats']),
        'source_tally': tally(ROUTEGEOM),
    }
    kin_den = np.concatenate([kd['kin'], kd['den']], 1)
    doc['sets']['Agent density + kin.'] = {
        'table_row': 'Agent density + kin.',
        'file': str(REPO / 'data' / 'b2d' / 'baseline_kin_den.npz'),
        'shape': [int(kd['kin'].shape[0]), 18], 'dtype': str(kd['kin'].dtype),
        'names_format': "<id> (bare, keys 'kin_names' / 'den_names'; concatenated kin||den at load time "
                        'by run_us.py:load_descriptor_arms)',
        'producer': '/home/jeongtae/SCIRT/b2d_irt/extract_baseline_feats.py (lines 16-40)',
        'producer_status': 'present in the repo, reads the raw anno directly; the third block it writes '
                           '(dino, 1024d) is not used by Table 3A and is not in the shipped npz',
        'published_method': None,
        'shipped_layout_string': None,
        'verified': 'the documented formulas were re-run from the raw anno for 4 routes '
                    '(10857, 11381, 11715, 11755): bit-for-bit identical to the shipped npz.',
        'columns': block(KIN + DEN, kin_den),
        'source_tally': tally(KIN + DEN),
        'notes': ['the heading used by dheading_mean/max comes from the ego position increments '
                  '(atan2 of diff x, diff y), not from the logged theta, so it is undefined-noisy while '
                  'the probe is stopped',
                  'no frame-validity mask is applied: every annotated frame of the rollout is used, '
                  'and routes with fewer than 5 annotation files are skipped entirely by the producer',
                  'duration_s is the rollout length, i.e. it encodes how long PDM-Lite needed / how '
                  'early the route terminated -- the most explicitly probe-dependent column of the set'],
    }
    doc['sets']['Kinematics (cmdkin, 25d)'] = {
        'table_row': 'Kinematics (cmdkin, 25d)',
        'file': str(REPO / 'data' / 'features' / 'eval_cmdkin_stats.npz'),
        'shape': list(ck['stats'].shape), 'dtype': str(ck['stats'].dtype),
        'names_format': 'route_<id>',
        'producer': 'ORIGINAL PRODUCER MISSING from every repo; reconstructed at '
                    'experiments/us_official/cmdkin.py:cmdkin and verified bit-for-bit (all 220 x 25 '
                    'float32 values equal) against the shipped npz, recomputed from the raw anno',
        'producer_status': 'reconstructed 2026-09; the two surviving slices '
                           '(eval_cmdkin_kinonly.npz = cols 0:12, eval_cmdkin_cmdonly.npz = cols 12:25) '
                           'are cut from this file by b2d_irt/scripts/cmdkin_decompose.py',
        'published_method': None,
        'shipped_layout_string': str(ck['layout']),
        'reconstruction': {
            'script': '/home/jeongtae/SC-IRT/experiments/us_official/cmdkin.py',
            'output': '/data2/jeongtae/official_baselines/us_features/inhouse_cmdkin.npz',
            'bit_for_bit': True,
            'columns_not_reproduced': [],
            'numeric_detail': 'all statistics are evaluated on the float32 ego/cmd arrays produced by '
                              'the eval_pipeline.load_anno cast; heading differences are wrapped with '
                              'the float32 modulo form ((dh + pi) %% (2 pi) - pi). Any other wrap '
                              '(np.angle(exp(i dh)), np.unwrap, arctan2, or the same modulo in float64) '
                              'reproduces yaw_mean / yaw_max only to about 1.8e-07, i.e. not bit-for-bit; '
                              'computing the whole block in float64 breaks 9 of the 25 columns at the '
                              'float32 ULP level.',
            'scored': 'AUROC 0.7519106345648979 / scene-MAE 0.18029126621966515 / rho 0.49723177046454564 '
                      '-- identical to 16 digits to the shipped "Kinematics (cmdkin)" row '
                      '(experiments/us_official/score_one.py).',
        },
        'verified': 'all 220 x 25 float32 values regenerated from the raw anno by '
                    'experiments/us_official/cmdkin.py are bit-for-bit identical to the shipped npz, '
                    'and the rebuilt file scores identically to 16 digits in Table 3A.',
        'columns': block(CMDKIN, ck['stats']),
        'source_tally': tally(CMDKIN),
    }
    doc['sets']['Hand-crafted risk (cmdkin+gtrisk, 73d)'] = {
        'table_row': 'Hand-crafted risk (cmdkin+gtrisk, 73d)',
        'file': 'concatenation of ' + str(REPO / 'data' / 'features' / 'eval_cmdkin_stats.npz') +
                ' (25d, cols 0-24) and ' + str(REPO / 'data' / 'features' / 'eval_gtrisk.npz') + ' (48d, cols 25-72)',
        'shape': [int(gt['stats'].shape[0]), 73], 'dtype': str(gt['stats'].dtype),
        'names_format': 'route_<id>',
        'producer': 'cmdkin block: see above. gtrisk block: '
                    '/home/jeongtae/SCIRT/b2d_irt/scripts/gt_risk.py (step_risk / step_static / '
                    'scene_series / agg), run --split eval',
        'producer_status': 'gt_risk.py present in the repo and reads the raw anno directly '
                           '(every 5th frame, 2 Hz; 60 m agent radius)',
        'published_method': None,
        'shipped_layout_string': str(gt['layout']),
        'parameters': {'GRID': 5, 'V0': 8.0, 'HOR': 4.0, 'R_MAX': 60.0, 'FOV_deg': 60.0,
                       'MARGIN_m': 0.3, 'bootstrap_m': 20, 'bootstrap_reps': 128, 'bootstrap_seed': 0,
                       'ttc_floor_s': 0.5, 'gap_floor_m': 0.5},
        'verified': 'gtrisk block re-run from the raw anno for 8 routes with '
                    'gt_risk.agg(gt_risk.scene_series(...)): bit-for-bit identical to the shipped npz. '
                    'cmdkin block: full 220-route bit-for-bit reconstruction, see the set above.',
        'columns': [{'index': j, 'name': 'cmdkin_' + c['name'], 'formula': c['formula'],
                     'input': c['input'], 'source': c['source'], 'sanity': c['sanity']}
                    for j, c in enumerate(block(CMDKIN, ck['stats']))]
                   + [{'index': 25 + j, 'name': 'gtrisk_' + c['name'], 'formula': c['formula'],
                       'input': c['input'], 'source': c['source'], 'sanity': c['sanity']}
                      for j, c in enumerate(block(GTRISK, gt['stats']))],
        'source_tally': {k: tally(CMDKIN).get(k, 0) + tally(GTRISK).get(k, 0)
                         for k in set(tally(CMDKIN)) | set(tally(GTRISK))},
    }

    doc['source_taxonomy'] = SRC
    doc['probe_vs_scene_accounting'] = {
        'question': 'per set, how many dimensions come from PDM-Lite own ego motion versus the '
                    'surrounding scene',
        'Route geometry (16d)': {
            'probe_ego_motion_only': 1, 'probe_route_command_only': 7, 'mixed_ego_x_command': 8,
            'other_agents': 0,
            'reading': 'all 16 dimensions are read off the PDM-Lite rollout log; zero dimensions see '
                       'any other agent. The 7 "route command" dimensions describe the nav route rather '
                       'than the probe motion, but they are still rollout-conditioned (the waypoint '
                       'polyline only extends as far as PDM-Lite drove, and n_cmd_seg counts frames).'},
        'Agent density + kin. (18d)': {
            'probe_ego_motion_only': 10, 'other_agents_ego_independent': 4,
            'other_agents_at_probe_pose': 2, 'other_agents_at_probe_pose_and_speed': 2,
            'reading': '10 of 18 are pure probe motion (including duration_s, which is literally how '
                       'long PDM-Lite took). Of the 8 scene dimensions only 4 (n_agents_mean/max, '
                       'n_walker_mean, frac_frames_with_walker) are ego-independent; the other 4 are '
                       'measured relative to the probe pose (10 m radius) or its speed.'},
        'Kinematics (cmdkin, 25d)': {
            'probe_ego_motion_only': 12, 'probe_route_command_only': 13, 'other_agents': 0,
            'reading': 'all 25 dimensions come from the PDM-Lite rollout and none sees another agent. '
                       'The 13 command dimensions are TIME-weighted fractions, so probe dwell time '
                       '(55% of the eval steps are a stopped probe) enters them directly.'},
        'Hand-crafted risk (cmdkin+gtrisk, 73d)': {
            'probe_ego_motion_only': 12, 'probe_route_command_only': 13,
            'other_agents_at_probe_pose': 30, 'other_agents_at_probe_pose_and_speed': 9,
            'other_agents_at_probe_pose_counterfactual_speed': 9,
            'reading': '25 of 73 are the probe-only cmdkin block. All 48 gtrisk dimensions involve other '
                       'agents but every one of them is evaluated in the probe rollout ego frame; 9 also '
                       'use the probe realized speed and 9 replace it with 8 m/s. Zero dimensions are '
                       'independent of the probe.'},
        'paper_wording': 'these four rows are probe-rollout descriptors, not scene-only descriptors',
    }

    doc['ssm_conformance'] = {
        'reference': 'B. Westhofen, C. Neurohr, T. Koopmann, M. Butz, B. Boede, F. Reichenbaecher, '
                     'M. Bollmann, R. Schuldes, J. Hoffmann, "Criticality Metrics for Automated Driving: '
                     'A Review and Suitability Analysis of the Available Metrics", arXiv:2108.02403 '
                     '(Artificial Intelligence Review, 2023); TTC/DRAC/PET definitions in the '
                     'time-scale and deceleration-scale metric sections.',
        'scope': 'the surrogate-safety columns of the gtrisk block only '
                 '(b2d_irt/scripts/gt_risk.py:step_risk / step_static)',
        'channels': {
            'inv_ttc_vr': {
                'claimed': 'TTC',
                'conformant': 'partly',
                'why': 'It is a legitimate 2D constant-velocity TTC: first positive root of the '
                       'relative-motion overlap of two inflated elliptical footprints, minimised over '
                       'agents. Deviations from the standard definition: (a) the ego is propagated '
                       'STRAIGHT along its current heading, its yaw rate is ignored, so TTC is wrong on '
                       'exactly the turning conflicts the bank is full of; (b) footprints are ellipses '
                       'inflated by 0.3 m rather than rectangles, which makes overlap optimistic at the '
                       'corners; (c) the value stored is 1 / max(TTC, 0.5 s), a monotone but saturating '
                       'transform, so every conflict below 0.5 s is indistinguishable (cap 2.0); '
                       '(d) already-overlapping agents get TTC = 0 -> the cap value.'},
            'inv_ttc_v0': {
                'claimed': 'TTC',
                'conformant': 'no',
                'why': 'the ego speed is replaced by a nominal 8 m/s, so it is not the TTC of the '
                       'observed scene but of a counterfactual constant-speed ego on the probe path. '
                       'It is a what-if criticality index, not a surrogate safety measure of the log.'},
            'rdec_vr': {
                'claimed': 'DRAC (deceleration rate to avoid a crash)',
                'conformant': 'yes, with two caveats',
                'why': 'the formula dv^2 / (2 gap) is exactly the standard DRAC. Caveats: the leader set '
                       'is chosen by a rectangular corridor test in the ego frame (|lateral| < combined '
                       'half-width + 0.3 m, longitudinal > 0) rather than by lane assignment, and the gap '
                       'is floored at 0.5 m so DRAC saturates instead of diverging on contact; the scene '
                       'value is the max over in-corridor agents (a scene-level aggregation of a '
                       'pairwise metric, which is standard practice).'},
            'rdec_v0': {
                'claimed': 'DRAC',
                'conformant': 'no',
                'why': 'same counterfactual 8 m/s ego speed as inv_ttc_v0; DRAC is defined on the '
                       'observed closing speed.'},
            'n_conflict_vr / n_conflict_v0': {
                'claimed': 'conflict count',
                'conformant': 'n/a (not a named SSM)',
                'why': 'counts agents whose constant-velocity closest approach inside a 4 s horizon '
                       'falls inside the inflated ego footprint. This is a thresholded '
                       'distance-of-closest-approach / predicted-minimum-distance quantity, not TTC, '
                       'DRAC or PET. Reporting it as "conflicts" is fine as long as the paper does not '
                       'call it an SSM.'},
            'inv_gap': {'claimed': 'proximity', 'conformant': 'n/a',
                        'why': 'inverse minimum surface-to-surface distance; a distance-scale proximity '
                               'index, not one of the named SSMs.'},
            'closing_max': {'claimed': 'range rate', 'conformant': 'n/a',
                            'why': 'positive range rate towards a STATIONARY ego. It is a legitimate '
                                   'probe-free approach-rate channel but it is not TTC (no size, no '
                                   'collision condition).'},
        },
        'PET': {
            'present': False,
            'why': 'no post-encroachment time is computed anywhere in gtrisk. PET requires the times at '
                   'which two road users successively occupy a shared conflict area, which needs a '
                   'conflict-area definition and no motion prediction; the gtrisk conflict channels are '
                   'prediction-based instead. If the paper text claims a PET baseline, that claim is '
                   'unsupported by this feature file.'},
        'summary': 'Of the standard trio, DRAC is implemented conformantly (in the _vr variant), TTC is '
                   'implemented as a straight-line constant-velocity 2D approximation with a 0.5 s floor '
                   'and an inverted scale, and PET is absent. The _v0 duplicates of TTC and DRAC are '
                   'counterfactual and should not be described as surrogate safety measures of the '
                   'recorded scene.',
    }

    doc['known_defects_inherited_from_the_producers'] = [
        "gt_risk.py's own layout string records that the '_v0 is probe-free' claim was refuted on "
        'review: the ego POSE is still the probe rollout pose (about 55% of eval steps are a stopped '
        'probe), and the 39-d "probe-free" subset falls to rho +0.072 after residualising on cmdkin.',
        'cmdkin_decompose.py documents that the cmdkin command block is TIME-weighted, so a probe that '
        'dwells inflates whichever command it was in; that is why the distance-weighted routegeom arm '
        'exists.',
        'baseline_kin_den den channels apply no class filter and no radius filter to the bounding boxes; '
        'in the 220 eval routes the only classes present are vehicle / walker / ego_vehicle, so the count '
        'is effectively "all vehicles and walkers logged in the frame", however far away.',
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(OUT, 'w'), indent=1)
    print('wrote', OUT)

    # per-npz provenance for the one file this family rebuilt
    side = OUT.parent / 'inhouse_cmdkin_provenance.json'
    json.dump({'npz': '/data2/jeongtae/official_baselines/us_features/inhouse_cmdkin.npz',
               'full_record': str(OUT),
               **{k: doc[k] for k in ('written', 'atdrive_commit', 'python', 'numpy', 'how_scored')},
               **doc['sets']['Kinematics (cmdkin, 25d)']}, open(side, 'w'), indent=1)
    print('wrote', side)
    for k, v in doc['sets'].items():
        print(f'  {k}: {len(v["columns"])} columns documented, tally {v["source_tally"]}')


if __name__ == '__main__':
    main()
