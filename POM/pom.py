import numpy as np
from copy import deepcopy

mat = np.zeros((3, 3, 6))
mat[2, 0, 1]=-1
mat[2, 0, 4]=1
mat[2, 1, 0]=1
mat[2, 1, 3]=-1
mat[1, 0, 2]=1
mat[1, 0, 5]=-1
mat[1, 2, 0]=-1
mat[1, 2, 3]=1
mat[0, 1, 2]=1
mat[0, 1, 5]=-1
mat[0, 2, 1]=-1
mat[0, 2, 4]=1

id_map = {
    (1,0,1): [None, 1, 4],
    (1,1,0): [None, 2, 5],
    (0,1,1): [None, 0, 3]
}

def validate_position(x, y, z, node_id=None):
    err = 'Selected node does not exist.'
    x, y, z = x % 2, y % 2, z % 2

    if x + y + z != 2:
        raise ValueError(err)

    if node_id not in id_map[x,y,z]:
        raise ValueError(err)


# build connection map
def check_servo_mappings(s, ee_pos):
    connected_servos=[]
    for servo in s:
        if servo.pos[servo.act_dim] == ee_pos[servo.act_dim]:
            connected_servos.append(servo)
    return(connected_servos)

class ee:
    def __init__(self, x, y, z, node_id, s):
        validate_position(x, y, z, node_id)
        self.pos=(x, y, z)
        self.node_id = node_id
        self.disp=np.zeros((3))
        self.control = check_servo_mappings(s, self.pos)

class servo:
    def __init__(self, x, y, z, act_dim, gait):
        validate_position(x, y, z)
        self.pos = (x, y, z)
        self.act_dim=act_dim  # 1 = 0134, 2=0235, 0=1245
        if act_dim==0:
            self.connected=z+2
        elif act_dim==1:
            self.connected=y+2
        elif act_dim==2:
            self.connected=x+2
        self.gait=gait
        self.disp = 0

def sim(e, *t):
    simdisp=[]

    if t !=():
        for i,end in enumerate(e):
            for time in t[0]:
                end.disp=0
                for controls in end.control:
                    controls.disp = controls.gait[time]
                    # calculate displacements as ind sum of components
                    end.disp += controls.disp*mat[controls.act_dim, :, end.node_id]
                if end.control==[]: # if no connected servos
                    simdisp.append(np.zeros((3)))
                else:
                    simdisp.append(deepcopy(end.disp))
    else:

        for i,end in enumerate(e):
            end.disp=0
            for controls in end.control:
                controls.disp = controls.gait
                # calculate displacements as ind sum of components
                end.disp += controls.disp*mat[controls.act_dim, :, end.node_id]
                # CHECKERBOARD PARITY, displacement direction:
                for i in range(3):
                    #if movement in dim, check dir:
                    if controls.disp*mat[controls.act_dim, :, end.node_id][i]!=0.:
                        kEnd=0
                        kControl=0
                        if end.pos[i]%4 in (0, 1):
                            kEnd=1
                        if (controls.pos[i]%4) in (0, 1):
                            kControl=1
                        if kEnd!=kControl:
                            end.disp[i]*=-1

            if end.control==[]: # if no connected servos
                simdisp.append(np.zeros((3)))
            else:
                simdisp.append(deepcopy(end.disp))

    return np.array(simdisp)



class TestCases:
    def test_ex1(self):
        s1 = [servo(2, 1, 1, 1, -5)]
        ees = [ee(0, 1, 3, 0, s1),
               ee(1, 1, 4, 5, s1)]
        desired = np.array([[0., 0., -5.],
                            [-5., 0., 0.]])
        assert (sim(ees) == desired).all()

    def test_position_validation(self):
        import pytest
        with pytest.raises(ValueError):
            ee(2, 1, 1, 1, [])
        with pytest.raises(ValueError):
            ee(2, 1, 1, 2, [])

    def test_ex1_variant(self):
        '''
        Any given node has two equally valid IDs. Make sure the sign
        is the same if you choose the other in both cases.
        '''
        s1 = [servo(2, 1, 1, 1, -5)]
        ees = [ee(0, 1, 3, 3, s1),
               ee(1, 1, 4, 2, s1)]
        desired = np.array([[0., 0., -5.],
                            [-5., 0., 0.]])
        assert (sim(ees) == desired).all()
