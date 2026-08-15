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
