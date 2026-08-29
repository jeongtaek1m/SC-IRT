"""The training scripts must at least parse: they are not imported by the
package, so a syntax error would otherwise reach users before any test."""
import glob
import os

def test_train_scripts_compile():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scripts = glob.glob(os.path.join(root, "train", "*.py")) + \
              glob.glob(os.path.join(root, "experiments", "*.py"))
    assert scripts
    for p in scripts:
        compile(open(p).read(), p, "exec")


def test_trainer_has_no_stale_ablation_attributes():
    """Removing an argparse flag must remove every a.<flag> reference with it."""
    import ast
    import re

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "train", "train_encoder_unified.py"), encoding="utf-8").read()
    declared = set(re.findall(r'add_argument\(["\']--(\w+)["\']', src))
    used = {n.attr for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "a"}
    assert used <= declared, f"undeclared argparse attributes: {sorted(used - declared)}"
