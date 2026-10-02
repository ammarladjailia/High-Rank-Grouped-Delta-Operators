import tempfile
import os
from experiments.toy_dynamics import run_toy_dynamics

def test_toy_dynamics_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_toy_dynamics(tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "active_subspace_dynamics.png"))
        assert os.path.exists(os.path.join(tmpdir, "full_space_decomposition.png"))
        assert os.path.exists(os.path.join(tmpdir, "toy_results.csv"))
        assert os.path.exists(os.path.join(tmpdir, "toy_summary.json"))
