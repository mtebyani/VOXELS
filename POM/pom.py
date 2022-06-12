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
    X, Y, Z = 0, 1, 2

    def __init__(self, node, direction):
        assert direction in range(3)

        self.node = node
        self.other_node = Node(*node.pos, node.nid)
        self.other_node.pos[direction] -= 2

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
        self._c_id = A[self.pom, direction, self.node.nid]

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
            c = (-1)**abs((node.pos - self.node.pos).sum() // 2)
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
        '''
        Return the positions of the end effectors when the servos are
        actuated by the given amounts. Positions are returned in an
        array of shape (N,3) where N is the number of end effectors.
        '''
        assert len(amounts) == len(self.servos)

        disp = []
        for e,effector in enumerate(self.effectors):
            disp_e = np.zeros(3)
            for amount,servo in zip(amounts, self.servos):
                disp_e += servo.actuate(effector, amount)
            disp.append(disp_e)
        return np.array(disp)

    def simulate(self, gait):
        '''
        Given an array of shape (M,T) with the positions of M servos
        at T different timesteps, return an array of N end effector
        positions at each of those times with shape (N,3,T).
        '''
        return np.array([self.actuate(*amounts).T
                         for amounts in np.transpose(gait)]).T


class VoxelBot(Voxels):
    def __init__(self):
        super().__init__()

        self.add_servo(Node(2, 1, 1), Servo.Z)
        self.add_servo(Node(1, 2, 1), Servo.Z)
        self.add_servo(Node(2, 3, 1), Servo.Z)
        self.add_servo(Node(3, 2, 1), Servo.Z)

        self.add_effector(Node(1, 3, 4))
        self.add_effector(Node(3, 3, 4))
        self.add_effector(Node(1, 1, 4))
        self.add_effector(Node(3, 1, 4))


class TestVoxelBotGait:
    def test_step1(self):
        v = VoxelBot()
        desired = np.array([[0, 5, 0],
                            [0, 0, 0],
                            [5, -5, 0],
                            [-5, 0, 0]])
        assert np.all(v.actuate(5, -5, 0, 0) == desired)

    def test_step2(self):
        v = VoxelBot()
        desired = np.array([[5, 5, 0],
                            [-5, 5, 0],
                            [5, -5, 0],
                            [-5, -5, 0]])
        assert np.all(v.actuate(5, -5, 5, -5) == desired)

    def test_step3(self):
        v = VoxelBot()
        desired = np.array([[5, 0, 0],
                            [-5, 5, 0],
                            [0, 0, 0],
                            [0, -5, 0]])
        assert np.all(v.actuate(0, 0, 5, -5) == desired)

    def test_step4(self):
        v = VoxelBot()
        desired = np.array([[5, -5, 0],
                            [-5, 5, 0],
                            [-5, 5, 0],
                            [5, -5, 0]])
        assert np.all(v.actuate(-5, 5, 5, -5) == desired)

    def test_step5(self):
        v = VoxelBot()
        desired = np.array([[0, -5, 0],
                            [0, 0, 0],
                            [-5, 5, 0],
                            [5, 0, 0]])
        assert np.all(v.actuate(-5, 5, 0, 0) == desired)

    def test_step6(self):
        v = VoxelBot()
        desired = np.array([[-5, -5, 0],
                            [5, -5, 0],
                            [-5, 5, 0],
                            [5, 5, 0]])
        assert np.all(v.actuate(-5, 5, -5, 5) == desired)

    def test_step7(self):
        v = VoxelBot()
        desired = np.array([[-5, 0, 0],
                            [5, -5, 0],
                            [0, 0, 0],
                            [0, 5, 0]])
        assert np.all(v.actuate(0, 0, -5, 5) == desired)

    def test_step8(self):
        v = VoxelBot()
        desired = np.array([[-5, 5, 0],
                            [5, -5, 0],
                            [5, -5, 0],
                            [-5, 5, 0]])
        assert np.all(v.actuate(5, -5, -5, 5) == desired)
