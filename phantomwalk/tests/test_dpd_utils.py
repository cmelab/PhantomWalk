import pytest
import numpy as np
import gsd.hoomd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'lib')))
import dpd_utils
from dpd_utils import initialize_snapshot_rand_walk

@pytest.fixture
def frame():
    return initialize_snapshot_rand_walk(num_pol=5, num_mon=10,density=0.8)
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
    computed_density = (num_pol * num_mon) / (L**3)
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
    f1 = initialize_snapshot_rand_walk(num_pol=3, num_mon=5, density=0.8, seed=99)
    f2 = initialize_snapshot_rand_walk(num_pol=3, num_mon=5, density=0.8, seed=99)
    np.testing.assert_array_equal(f1.particles.position, f2.particles.position)

def test_different_seeds_give_different_positions():
    f1 = initialize_snapshot_rand_walk(num_pol=3, num_mon=5, density=0.8, seed=1)
    f2 = initialize_snapshot_rand_walk(num_pol=3, num_mon=5, density=0.8, seed=2)
    assert not np.allclose(f1.particles.position, f2.particles.position)

def test_bond_lengths_are_correct():
    bond_length = 1.0
    num_pol=5
    num_mon=50
    snap = initialize_snapshot_rand_walk(num_pol=num_pol, num_mon=num_mon, density=1.0, bond_length=bond_length)
    L = snap.configuration.box[0]
    frame_ds = []
    for j in range(num_pol):
        idx = j*num_mon
        d1 = snap.particles.position[idx:idx+num_mon-1] - snap.particles.position[idx+1:idx+num_mon]
        L = snap.configuration.box[0]
        d1 -= L*np.round(d1/L)
        bond_l = np.linalg.norm(d1,axis=1)
        frame_ds.append(bond_l)
    avg_frame_bond_l = np.mean(np.array(frame_ds))
    assert (bond_length - 0.01) <= avg_frame_bond_l <= (bond_length + 0.01)
