import pytest
import numpy as np
import gsd.hoomd

import sys
import os
sys.path.append('../lib/')
import dpd_utils
from dpd_utils import initialize_snapshot_rand_walk

@pytest.fixture
def frame():
    return initialize_snapshot_rand_walk(num_pol=5, num_mon=10)
''' TODO: update code for non-cubic boxes and update this section
def test_box_is_cubic(frame):
    box = default_frame.configuration.box
    assert box[0] == box[1] == box[2]

def test_box_tilt_factors_zero(frame):
    box = default_frame.configuration.box
    assert box[3] == box[4] == box[5] == 0
'''

def test_box_volume_matches_density():
    num_pol, num_mon, density = 10, 20, 0.85
    frame = initialize_snapshot_rand_walk(num_pol, num_mon, density=density)
    L = frame.configuration.box[0]
    computed_density = (num_pol * num_mon) / L**3
    assert computed_density == pytest.approx(density, rel=1e-5)

def test_positions_inside_box(frame):
    L = frame.configuration.box[0]
    pos = frame.particles.position
    assert np.all(pos >= -L / 2)
    assert np.all(pos <   L / 2)

def test_bond_count(frame):
    num_pol, num_mon = 5, 10
    assert frame.bonds.N == num_pol * (num_mon - 1)

#TODO add counts for angles and dihedrals
#TODO add code and tests for non-linear and polydisperse systems

def test_seed_reproducibility():
    f1 = initialize_snapshot_rand_walk(num_pol=3, num_mon=5, seed=99)
    f2 = initialize_snapshot_rand_walk(num_pol=3, num_mon=5, seed=99)
    np.testing.assert_array_equal(f1.particles.position, f2.particles.position)

def test_different_seeds_give_different_positions():
    f1 = initialize_snapshot_rand_walk(num_pol=3, num_mon=5, seed=1)
    f2 = initialize_snapshot_rand_walk(num_pol=3, num_mon=5, seed=2)
    assert not np.allclose(f1.particles.position, f2.particles.position)

def test_bond_lengths_are_correct():
    """With per-step PBC wrapping, raw distances should equal bond_length directly."""
    bond_length = 1.0
    frame = initialize_snapshot_rand_walk(num_pol=5, num_mon=10, bond_length=bond_length)
    pos = frame.particles.position
    for a, b in frame.bonds.group:
        dist = np.linalg.norm(pos[b] - pos[a])
        assert dist == pytest.approx(bond_length, rel=1e-5)
