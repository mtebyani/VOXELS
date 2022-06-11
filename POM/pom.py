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

class Servo:
    def __init__(self, node, direction):
        assert direction in range(3)

        self.node = node

        # Unfortunately we can only identify the correct plane of
        # motion by elimination: it can't be the plane where the
        # actuator moves or the plane where it's on a voxel boundary.
        # This is always unique because it's physically impossible to
        # rotate the actuator to make those two planes the same (this
        # will raise a ValueError here).
        planes = np.ones(3, bool)
        planes[direction] = False
        planes[self.node.id % 3] = False
        (self.pom,) = np.nonzero(planes)[0]

        # Correction to the sign for when this servo rotates the plane
        # of motion "backwards".
        self._c_id = A[:, direction, self.node.id].sum()

    def conflicts_with(self, other):
        'Return whether this servo shares a plane of motion with another.'
        return (self.pom == other.pom
                and self.node.pos[self.pom] == other.node.pos[other.pom])

    def actuate(self, node, amount):
        '''
        Return how much this servo affects the specified node when
        actuated by a given amount.
        '''
        if node.pos[self.pom] == self.node.pos[self.pom]:
            c_pos = (-1)**abs((self.node.pos - node.pos).sum())
            return self._c_id * c_pos * amount * A[self.pom, :, node.id]
        return np.zeros(3)


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
            for amount,servo in zip(amounts, self.servos):
                disp_e += servo.actuate(effector, amount)
            disp.append(disp_e)
        return np.array(disp)


class TestNodeInvariants:
    def test_node_ids_must_be_0_to_5(self):
        from pytest import raises
        with raises(AssertionError):
            Node(1, 2, 3, 6)

class TestServoInvariants:
    class TestInitializationValidation:
        def test_nodes_cant_move_in_their_fixed_direction(self):
            from pytest import raises
            with raises(ValueError):
                Servo(Node(0,0,0,0), 0)

            with raises(ValueError):
                Servo(Node(0,0,0,1), 1)

            with raises(ValueError):
                Servo(Node(0,0,0,2), 2)

            with raises(ValueError):
                Servo(Node(0,0,0,3), 0)

            with raises(ValueError):
                Servo(Node(0,0,0,4), 1)

            with raises(ValueError):
                Servo(Node(0,0,0,5), 2)

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
