# Copyright (C) 2025-2026 Changkai Zhang.
#
# This file is part of Alice library.
#
# Alice is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# Alice is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Alice. If not, see <https://www.gnu.org/licenses/>.


"""Tests for interaction map generation."""

import pytest
from alice.network.intrcmap import generate_snake_order, intrcmap_square


class TestSnakeOrder:
    """Tests for snake-like traversal order generation."""
    
    def test_2x3_lattice(self):
        """Test snake order for a 2x3 lattice."""
        ord_map, latt = generate_snake_order(lx=2, ly=3)
        
        # Expected snake order for 2x3 lattice (0-based):
        # 00-----05
        # |      |
        # 01     04
        # |      |
        # 02-----03
        
        expected_ord = [
            [0, 5],
            [1, 4],
            [2, 3]
        ]
        
        assert ord_map == expected_ord
        assert latt[0] == (0, 0)
        assert latt[3] == (2, 1)
        assert latt[5] == (0, 1)
    
    @pytest.mark.parametrize("lx,ly,expected_sites", [
        (2, 2, 4),
        (3, 2, 6),
        (4, 3, 12),
        (1, 5, 5),
    ])
    def test_total_sites(self, lx, ly, expected_sites):
        """Test that snake order generates correct number of sites."""
        ord_map, latt = generate_snake_order(lx, ly)
        assert len(latt) == expected_sites
        
        # Check that all sites are unique
        flat_ord = [site for row in ord_map for site in row]
        assert len(set(flat_ord)) == expected_sites
    
    def test_1d_chain(self):
        """Test snake order for 1D chain."""
        ord_map, latt = generate_snake_order(lx=5, ly=1)
        
        # 1D chain should be sequential
        assert ord_map == [[0, 1, 2, 3, 4]]
        assert latt == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]


class TestNearestNeighbor:
    """Tests for nearest-neighbor interactions."""
    
    def test_1d_chain(self, basic_1d_config):
        """Test 1D chain interactions."""
        interactions = intrcmap_square(basic_1d_config)
        
        lx = basic_1d_config['lx']
        # Should have (lx-1) NN interactions for 1D chain
        assert len(interactions) == lx - 1
        
        for i, intr in enumerate(interactions):
            assert intr.leading_site == i
            assert intr.terminal_site == i + 1
            assert 'NN' in intr.label
            assert 'N2Y' in intr.label
    
    def test_2d_square_lattice(self, basic_2d_config):
        """Test 2D square lattice with OBC."""
        interactions = intrcmap_square(basic_2d_config)
        
        lx = basic_2d_config['lx']
        ly = basic_2d_config['ly']
        
        # Expected NN interactions for OBC:
        # X-axis: ly * (lx - 1)
        # Y-axis: lx * (ly - 1)
        expected_x = ly * (lx - 1)
        expected_y = lx * (ly - 1)
        expected_total = expected_x + expected_y
        
        assert len(interactions) == expected_total
        
        x_interactions = [i for i in interactions if 'N2X' in i.label]
        y_interactions = [i for i in interactions if 'N2Y' in i.label]
        
        assert len(x_interactions) == expected_x
        assert len(y_interactions) == expected_y
    
    @pytest.mark.parametrize("lx,ly,expected_nn", [
        (2, 2, 4),   # 2x2: 2 X + 2 Y
        (3, 2, 7),   # 3x2: 4 X + 3 Y
        (4, 3, 17),  # 4x3: 9 X + 8 Y
    ])
    def test_nn_count_obc(self, lx, ly, expected_nn):
        """Test that NN interaction count is correct for various lattice sizes."""
        config = {
            'lx': lx, 'ly': ly,
            'bcx': 'OBC', 'bcy': 'OBC',
            'label': 'SpinSqLatt',
            'cpl': 1.0, 'cplp': [0.0, 0.0]
        }
        interactions = intrcmap_square(config)
        assert len(interactions) == expected_nn


class TestPeriodicBoundary:
    """Tests for periodic boundary conditions."""
    
    def test_pbc_y_cylinder(self, pbc_cylinder_config):
        """Test periodic boundary in Y direction (cylinder)."""
        interactions = intrcmap_square(pbc_cylinder_config)
        
        lx = pbc_cylinder_config['lx']
        
        pbc_interactions = [i for i in interactions if 'PBC' in i.label]

        # Should have lx PBC interactions (one per column)
        assert len(pbc_interactions) == lx

        # All PBC interactions should be in Y direction
        for intr in pbc_interactions:
            assert 'N2Y' in intr.label
    
    def test_pbc_x(self):
        """Test periodic boundary in X direction."""
        config = {
            'lx': 3, 'ly': 2,
            'bcx': 'PBC', 'bcy': 'OBC',
            'label': 'SpinSqLatt',
            'cpl': 1.0, 'cplp': [0.0, 0.0]
        }
        interactions = intrcmap_square(config)
        
        pbc_interactions = [i for i in interactions if 'PBC' in i.label]

        # Should have ly PBC interactions in X direction
        assert len(pbc_interactions) == 2

        for intr in pbc_interactions:
            assert 'N2X' in intr.label
    
    def test_pbc_both_torus(self, torus_config):
        """Test periodic boundary in both directions (torus)."""
        interactions = intrcmap_square(torus_config)
        
        pbc_interactions = [i for i in interactions if 'PBC' in i.label]

        # Should have PBC in both X and Y
        pbc_x = [i for i in pbc_interactions if 'N2X' in i.label]
        pbc_y = [i for i in pbc_interactions if 'N2Y' in i.label]
        
        assert len(pbc_x) > 0
        assert len(pbc_y) > 0


class TestNextNearestNeighbor:
    """Tests for next-nearest-neighbor (J2) interactions."""
    
    def test_j2_interactions(self, j2_config):
        """Test J2 (NNN) interactions."""
        interactions = intrcmap_square(j2_config)
        
        lx = j2_config['lx']
        ly = j2_config['ly']
        
        # Count NNN interactions
        nnn_d = [i for i in interactions if 'N3D' in i.label]
        nnn_o = [i for i in interactions if 'N3O' in i.label]

        # For lattice with OBC:
        # N3D: (Lx-1) * (Ly-1)
        # N3O: (Lx-1) * (Ly-1)
        expected_nnn = (lx - 1) * (ly - 1)

        assert len(nnn_d) == expected_nnn
        assert len(nnn_o) == expected_nnn

        # Check coupling values
        for intr in nnn_d:
            assert intr.cpl == 0.5
        for intr in nnn_o:
            assert intr.cpl == 0.3
    
    def test_j2_only_diagonal(self):
        """Test with only diagonal J2 interactions."""
        config = {
            'lx': 3, 'ly': 3,
            'bcx': 'OBC', 'bcy': 'OBC',
            'label': 'SpinSqLatt',
            'cpl': 1.0, 'cplp': [0.5, 0.0]  # Only diagonal
        }
        interactions = intrcmap_square(config)
        
        nnn_d = [i for i in interactions if 'N3D' in i.label]
        nnn_o = [i for i in interactions if 'N3O' in i.label]

        assert len(nnn_d) == 4
        assert len(nnn_o) == 0

    def test_j2_only_off_diagonal(self):
        """Test with only off-diagonal J2 interactions."""
        config = {
            'lx': 3, 'ly': 3,
            'bcx': 'OBC', 'bcy': 'OBC',
            'label': 'SpinSqLatt',
            'cpl': 1.0, 'cplp': [0.0, 0.3]  # Only off-diagonal
        }
        interactions = intrcmap_square(config)

        nnn_d = [i for i in interactions if 'N3D' in i.label]
        nnn_o = [i for i in interactions if 'N3O' in i.label]
        
        assert len(nnn_d) == 0
        assert len(nnn_o) == 4
    
    def test_j2_with_pbc(self):
        """Test J2 interactions with PBC."""
        config = {
            'lx': 3, 'ly': 3,
            'bcx': 'PBC', 'bcy': 'PBC',
            'label': 'SpinSqLatt',
            'cpl': 1.0, 'cplp': [0.5, 0.3]
        }
        interactions = intrcmap_square(config)
        
        # Count J2 PBC interactions
        j2_pbc = [i for i in interactions if 'NNN' in i.label and 'PBC' in i.label]
        
        # Should have J2 PBC interactions
        assert len(j2_pbc) > 0


class TestInteractionProperties:
    """Tests for general interaction properties."""
    
    def test_sorting(self, torus_config):
        """Test that interactions are sorted by leading_site."""
        interactions = intrcmap_square(torus_config)

        # Check that leading_site values are non-decreasing
        for i in range(len(interactions) - 1):
            assert interactions[i].leading_site <= interactions[i + 1].leading_site
    
    def test_coupling_values(self):
        """Test that coupling values are correctly assigned."""
        config = {
            'lx': 2, 'ly': 2,
            'bcx': 'OBC', 'bcy': 'OBC',
            'label': 'SpinSqLatt',
            'cpl': 2.5,
            'cplp': [0.0, 0.0]
        }
        interactions = intrcmap_square(config)
        
        # All NN interactions should have cpl = 2.5
        for intr in interactions:
            if 'NN' in intr.label:
                assert intr.cpl == 2.5
    
    def test_interaction_structure(self, basic_2d_config):
        """Test that each interaction has required fields."""
        interactions = intrcmap_square(basic_2d_config)
        
        for intr in interactions:
            assert hasattr(intr, 'leading_site')
            assert hasattr(intr, 'terminal_site')
            assert hasattr(intr, 'cpl')
            assert hasattr(intr, 'label')

            # leading_site should be less than terminal_site
            assert intr.leading_site < intr.terminal_site

            # label should be a list
            assert isinstance(intr.label, list)
            assert len(intr.label) > 0


class TestModelTypes:
    """Tests for different model types."""
    
    @pytest.mark.parametrize("label,cpl_key,cplp_key", [
        ('SpinSqLatt', 'j1', 'j2'),
        ('HubbardSqLatt', 't1', 't2'),
        ('SpinlessFerSqLatt', 't1', 't2'),
    ])
    def test_model_with_legacy_keys(self, label, cpl_key, cplp_key):
        """Test that legacy coupling keys (j1/j2, t1/t2) work."""
        config = {
            'lx': 2, 'ly': 2,
            'bcx': 'OBC', 'bcy': 'OBC',
            'label': label,
            cpl_key: 1.0,
            cplp_key: [0.0, 0.0]
        }
        interactions = intrcmap_square(config)
        
        # Should generate interactions without error
        assert len(interactions) > 0
        
        # All should have correct coupling value
        for intr in interactions:
            assert intr.cpl in [1.0, 0.0]
