import numpy as np

A = np.array([
    [[0., 0., 0., 0., 0., 0.],
     [0., 0., 1., 0., 0., -1],
     [0., -1, 0., 0., 1., 0.]],
    [[0., 0., -1, 0., 0., 1.],
     [0., 0., 0., 0., 0., 0.],
     [1., 0., 0., -1, 0., 0.]],
    [[0., 1., 0., 0., -1, 0.],
     [-1, 0., 0., 1., 0., 0.],
     [0., 0., 0., 0., 0., 0.]]])


class Node:
    def __init__(self, x, y, z, nid):
        assert nid in range(6)
        self.pos = np.array([x, y, z])
        self.id = nid

    def position(self):
        'Return the absolute position of the node.'
        axis_offset = 0.5*np.ones(3)
        axis_offset[self.id % 3] = self.id // 3
        return axis_offset + self.pos

    def _normalized_pos(self):
        '''
        Lattice position if we normalize node IDs to all be 0-3. This
        is used to check which nodes are in the same plane. It just
        treats all the higher node IDs as their lower-numbered
        counterpart in the next voxel over.
        '''
        return self.pos + (np.arange(3)==self.id % 3)*(self.id//3)

    def planes_match(self, other):
        'Return whether each of the three planes matches.'
        return self._normalized_pos() == other._normalized_pos()

class Servo:
    def __init__(self, node, direction):
        assert direction in range(3)

        self.node = node
        self._planes = np.arange(3) != direction

        # Correction to the sign for when this servo rotates the plane
        # of motion "backwards".
        self._c_id = A[:, direction, self.node.id].sum()

        if node.id % 3 == direction:
            DL = 'XYZ'[direction]
            raise ValueError(f'Servo at node {node} cannot actuate in {DL}')

    def conflicts_with(self, other):
        'Return whether this servo shares a plane of motion with another.'
        return np.any(self._planes & other._planes
                      & self.node.planes_match(other.node))

    def shared_planes(self, node):
        'Return the planes this Servo shares with some voxel position.'
        return np.nonzero(self.node.planes_match(node) & self._planes)[0]

    def actuate(self, node, amount):
        '''
        Return how much this servo affects the specified node when actuated by
        a given amount.
        '''
        ret = np.zeros(3)
        for p in self.shared_planes(node):
            c_pos = (-1)**abs((self.node.pos - node.pos).sum())
            ret += self._c_id * c_pos * amount * A[p, :, node.id]
        return ret


class Voxels:
    def __init__(self):
        self.servos = []
        self.effectors = []

    def add_servo(self, x, y, z, nid, direction):
        new = Servo(Node(x, y, z, nid), direction)
        for servo in self.servos:
            if servo.conflicts_with(new):
                raise ValueError('Servo makes system overdetermined.')
        self.servos.append(new)

    def add_effector(self, x, y, z, nid):
        self.effectors.append(Node(x, y, z, nid))

    def actuate(self, *amounts):
        assert len(amounts) == len(self.servos)

        disp = []
        for e,effector in enumerate(self.effectors):
            disp_e = np.zeros(3)
            for amount,(s,servo) in zip(amounts, enumerate(self.servos)):
                disp_e += servo.actuate(effector, amount)
            disp.append(disp_e)
        return np.array(disp)


class TestNodeInvariants:
    def test_node_ids_must_be_0_to_5(self):
        from pytest import raises
        with raises(AssertionError):
            Node(1, 2, 3, 6)

    def test_position_offsets_dont_depend_on_lattice_position(self):
        pos = np.random.randint(10, size=3)

        n = Node(*pos, 0)
        assert np.all(n.position() == pos+[0., 0.5, 0.5])

        n = Node(*pos, 1)
        assert np.all(n.position() == pos+[0.5, 0., 0.5])

        n = Node(*pos, 2)
        assert np.all(n.position() == pos+[0.5, 0.5, 0.])

        n = Node(*pos, 3)
        assert np.all(n.position() == pos+[1., 0.5, 0.5])

        n = Node(*pos, 4)
        assert np.all(n.position() == pos+[0.5, 1., 0.5])

        n = Node(*pos, 5)
        assert np.all(n.position() == pos+[0.5, 0.5, 1.])

    class TestSharedPlanesForSameID:
        def test_all_coords_different_means_no_planes_shared(self):
            for _ in range(100):
                nid = np.random.randint(6)
                pos = np.random.randint(10, size=3)
                offset = 1 + np.random.randint(8, size=3)
                offset[offset > 4] -= 9
                n1, n2 = Node(*pos, nid), Node(*pos+offset, nid)
                assert not np.any(n1.planes_match(n2))

        def test_one_coord_same_means_that_plane_shared(self):
            for _ in range(100):
                shared_plane = np.random.randint(3)
                nid = np.random.randint(6)
                pos = np.random.randint(10, size=3)
                offset = 1 + np.random.randint(8, size=3)
                offset[offset > 4] -= 9
                offset[shared_plane] = 0
                n1, n2 = Node(*pos, nid), Node(*pos+offset, nid)
                expected = np.arange(3) == shared_plane
                assert np.all(n1.planes_match(n2) == expected)

    class TestSharedPlanesForCompatibleID:
        def test_one_coord_offset_correctly_means_that_plane_shared(self):
            for _ in range(100):
                nid1 = np.random.randint(6)
                nid2 = nid1-3 if nid1 >= 3 else nid1+3
                shared_plane = nid1 % 3
                pos = np.random.randint(10, size=3)
                offset = 1 + np.random.randint(8, size=3)
                offset[offset > 4] -= 9
                offset[shared_plane] = 1 if nid1 >= 3 else -1
                n1, n2 = Node(*pos, nid1), Node(*pos+offset, nid2)
                expected = np.arange(3) == shared_plane
                assert np.all(n1.planes_match(n2) == expected)

        def test_one_coord_offset_wrong_means_no_planes_shared(self):
            for _ in range(100):
                nid1 = np.random.randint(6)
                nid2 = nid1-3 if nid1 >= 3 else nid1+3
                shared_plane = nid1 % 3
                pos = np.random.randint(10, size=3)
                offset = 1 + np.random.randint(8, size=3)
                offset[offset > 4] -= 9
                offset[shared_plane] = 42
                n1, n2 = Node(*pos, nid1), Node(*pos+offset, nid2)
                assert not np.any(n1.planes_match(n2))




class TestServoInvariants:
    class TestBaseNodeMovesByActuationAmount:

        def test_node_1_X(self):
            v = Voxels()
            v.add_servo(0, 0, 0, 1, 0)
            v.add_effector(0, 0, 0, 1)
            v.add_effector(0, -1, 0, 4)
            assert np.all(v.actuate(5) == [5, 0, 0])

        def test_node_5_X(self):
            v = Voxels()
            v.add_servo(0, 0, 0, 5, 0)
            v.add_effector(0, 0, 0, 5)
            v.add_effector(0, 0, 1, 2)
            assert np.all(v.actuate(5) == [5, 0, 0])

        def test_node_4_X(self):
            v = Voxels()
            v.add_servo(0, -1, 0, 4, 0)
            v.add_effector(0, 0, 0, 1)
            v.add_effector(0, -1, 0, 4)
            assert np.all(v.actuate(5) == [5, 0, 0])

        def test_node_2_X(self):
            v = Voxels()
            v.add_servo(0, 0, 1, 2, 0)
            v.add_effector(0, 0, 0, 5)
            v.add_effector(0, 0, 1, 2)
            assert np.all(v.actuate(5) == [5, 0, 0])

        def test_node_2_Y(self):
            v = Voxels()
            v.add_servo(0, 0, 0, 2, 1)
            v.add_effector(0, 0, 0, 2)
            v.add_effector(0, 0, -1, 5)
            assert np.all(v.actuate(5) == [0, 5, 0])

        def test_node_5_Y(self):
            v = Voxels()
            v.add_servo(0, 0, -1, 5, 1)
            v.add_effector(0, 0, 0, 2)
            v.add_effector(0, 0, -1, 5)
            assert np.all(v.actuate(5) == [0, 5, 0])

        def test_node_3_Y(self):
            v = Voxels()
            v.add_servo(0, 0, 0, 3, 1)
            v.add_effector(0, 0, 0, 3)
            v.add_effector(1, 0, 0, 0)
            assert np.all(v.actuate(5) == [0, 5, 0])

        def test_node_0_Y(self):
            v = Voxels()
            v.add_servo(1, 0, 0, 0, 1)
            v.add_effector(0, 0, 0, 3)
            v.add_effector(1, 0, 0, 0)
            assert np.all(v.actuate(5) == [0, 5, 0])

        def test_node_0_Z(self):
            v = Voxels()
            v.add_servo(0, 0, 0, 0, 2)
            v.add_effector(0, 0, 0, 0)
            v.add_effector(-1, 0, 0, 3)
            assert np.all(v.actuate(5) == [0, 0, 5])

        def test_node_3_Z(self):
            v = Voxels()
            v.add_servo(-1, 0, 0, 3, 2)
            v.add_effector(0, 0, 0, 0)
            v.add_effector(-1, 0, 0, 3)
            assert np.all(v.actuate(5) == [0, 0, 5])

        def test_node_4_Z(self):
            v = Voxels()
            v.add_servo(0, 0, 0, 4, 2)
            v.add_effector(0, 0, 0, 4)
            v.add_effector(0, 1, 0, 1)
            assert np.all(v.actuate(5) == [0, 0, 5])

        def test_node_1_Z(self):
            v = Voxels()
            v.add_servo(0, 1, 0, 1, 2)
            v.add_effector(0, 0, 0, 4)
            v.add_effector(0, 1, 0, 1)
            assert np.all(v.actuate(5) == [0, 0, 5])
