#!/usr/bin/env python3
"""drop_channel_run.py with --ablate-lane appended to the argv it replays.

WHY. The UPS table carries a second scene prior built from the speed-ablated
encoder. With R2-noLane the encoder of record, that control has to be lane-free
too, or the row moves two things at once. drop_channel_run.py reconstructs the
shipped command (drop_channel_run.py:332-337)

    r2_graph.py --domain b2d --gpu G --seed S [--epochs E] [--draws D]

and patches build_r2 so the chosen ego channels are zeroed in both ego paths.
This wrapper adds one thing: --ablate-lane on that argv. The channel zeroing,
the out_path redirect into --outdir with the --tag prefix, the drop_check
provenance and every frozen assert are drop_channel_run.py's own.

HOW. drop_channel_run.run() calls r2_graph.main() (line 339); we wrap that
function rather than the argv list, because the driver builds argv locally.
The output name is unaffected: bes.out_path is already redirected to
<outdir>/<tag>_<basename>, and r2_graph's own basename under --ablate-lane is
r2nolane_b2d_s<seed>.npz, so the file lands as <tag>_r2nolane_b2d_s<seed>.npz
and box['out'] tracks it.

    drop_channel_nolane.py --drop speed --seed 0 --gpu 0 \
        --outdir /data2/jeongtae/relgraph_e16sel/nolane_nospeed --tag nl
"""
import sys

sys.path.insert(0, '/data2/jeongtae/relgraph_e16sel')
import drop_channel_run as D                                        # noqa: E402

_orig_run = D.run


def run(a, names, idx, driver_cmd):
    import r2_graph
    _main = r2_graph.main

    def main():
        if '--ablate-lane' not in sys.argv:
            sys.argv = sys.argv + ['--ablate-lane']
        print(f'[drop_nolane] replayed argv {" ".join(sys.argv)}', flush=True)
        return _main()

    r2_graph.main = main
    try:
        return _orig_run(a, names, idx, driver_cmd + '  (+--ablate-lane)')
    finally:
        r2_graph.main = _main


D.run = run

if __name__ == '__main__':
    D.main()
