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
    def __init__(self, x, y, z, nid=None):
        self.pos = np.array([x, y, z])
        assert (self.pos % 2).sum() == 2

        # The user can specify their own node ID, but it is checked
        # for correctness.
        (self.nid,) = np.nonzero(self.pos % 2 == 0)[0]
        if nid is not None:
            assert nid in range(6) and nid % 3 == self.nid
            self.nid = nid

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
        planes[self.node.nid % 3] = False
        (self.pom,) = np.nonzero(planes)[0]

        # Correction to the sign for when this servo rotates the plane
        # of motion "backwards".
        self._c_id = A[:, direction, self.node.nid].sum()

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
            c_id = -1 if node.nid > 2 else 1
            c = (-1)**((node.pos - self.node.pos).sum() // 2)
            return c * c_id * self._c_id * amount * A[self.pom, :, node.nid]
        return np.zeros(3)


class Voxels:
    def __init__(self):
        self.servos = []
        self.effectors = []

    def add_servo(self, node, direction):
        new = Servo(node, direction)
        for servo in self.servos:
            if servo.conflicts_with(new):
                raise ValueError('Servo makes system overdetermined.')
        self.servos.append(new)

    def add_effector(self, node):
        self.effectors.append(node)

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
    def test_nodes_must_lie_on_grid(self):
        from pytest import raises
        with raises(AssertionError):
            Node(0,0,0)

    def test_nodes_from_later_tests_are_ok(self):
        Node(2, 1, 1)
        Node(0, 1, 3)
        Node(1, 1, 4)

    def test_node_ids_can_be_inferred(self):
        assert Node(0, 1, 3).nid == 0
        assert Node(1, 1, 4).nid == 2

    def test_node_ids_can_be_provided(self):
        Node(0, 1, 3, 3)
        Node(1, 1, 4, 5)

    def test_bad_node_ids_are_detected(self):
        from pytest import raises
        with raises(AssertionError):
            Node(0, 1, 3, 2)
        with raises(AssertionError):
            Node(1, 1, 4, 3)

class TestServoInvariants:
    class TestInitializationValidation:
        def test_nodes_cant_move_in_their_fixed_direction(self):
            from pytest import raises
            with raises(ValueError):
                Servo(Node(0,1,1,0), 0)

            with raises(ValueError):
                Servo(Node(1,0,1,1), 1)

            with raises(ValueError):
                Servo(Node(1,1,0,2), 2)


    class TestBaseNodeMovesByActuationAmount:
        def test_node_1_X(self):
            v = Voxels()
            n = Node(1, 0, 1)
            v.add_servo(n, 0)
            v.add_effector(n)
            assert np.all(v.actuate(5) == [5, 0, 0])

        def test_node_2_X(self):
            v = Voxels()
            n = Node(1, 1, 0)
            v.add_servo(n, 0)
            v.add_effector(n)
            assert np.all(v.actuate(5) == [5, 0, 0])

        def test_node_0_Y(self):
            v = Voxels()
            n = Node(0, 1, 1)
            v.add_servo(n, 1)
            v.add_effector(n)
            assert np.all(v.actuate(5) == [0, 5, 0])

        def test_node_2_Y(self):
            v = Voxels()
            n = Node(1, 1, 0)
            v.add_servo(n, 1)
            v.add_effector(n)
            assert np.all(v.actuate(5) == [0, 5, 0])

        def test_node_0_Z(self):
            v = Voxels()
            n = Node(0, 1, 1)
            v.add_servo(n, 2)
            v.add_effector(n)
            assert np.all(v.actuate(5) == [0, 0, 5])

        def test_node_1_Z(self):
            v = Voxels()
            n = Node(1, 0, 1)
            v.add_servo(n, 2)
            v.add_effector(n)
            assert np.all(v.actuate(5) == [0, 0, 5])


    class TestOtherNodeMovesByActuationAmount:
        def test_node_1_X(self):
            v = Voxels()
            v.add_servo(Node(1, 0, 1), 0)
            v.add_effector(Node(3, 0, 1))
            assert np.all(v.actuate(5) == [-5, 0, 0])

        def test_node_2_X(self):
            v = Voxels()
            v.add_servo(Node(1, 1, 0), 0)
            v.add_effector(Node(3, 1, 0))
            assert np.all(v.actuate(5) == [-5, 0, 0])

        def test_node_0_Y(self):
            v = Voxels()
            v.add_servo(Node(0, 1, 1), 1)
            v.add_effector(Node(0, 3, 1))
            assert np.all(v.actuate(5) == [0, -5, 0])

        def test_node_2_Y(self):
            v = Voxels()
            v.add_servo(Node(0, 1, 1), 1)
            v.add_effector(Node(0, 3, 1))
            assert np.all(v.actuate(5) == [0, -5, 0])

        def test_node_0_Z(self):
            v = Voxels()
            v.add_servo(Node(0, 1, 1), 2)
            v.add_effector(Node(0, 1, 3))
            assert np.all(v.actuate(5) == [0, 0, -5])

        def test_node_1_Z(self):
            v = Voxels()
            v.add_servo(Node(1, 0, 1), 2)
            v.add_effector(Node(1, 0, 3))
            assert np.all(v.actuate(5) == [0, 0, -5])
