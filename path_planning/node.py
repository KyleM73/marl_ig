class Node:
    def __init__(self):
        self.pose = (0,0)
        self.parent = None
        self.g, self.h, self.f = 0, 0, 0

    def set_pose(self, pose):
        self.pose = pose

    def get_pose(self):
        return self.pose

    def set_parent(self, parent):
        self.parent = parent

    def get_parent(self):
        return self.parent

    def set_g(self, g):
        self.g = g

    def get_g(self):
        return self.g

    def set_h(self, h):
        self.h = h

    def get_h(self):
        return self.h

    def set_f(self, f):
        self.f = f

    def get_f(self):
        return self.f

    def same_pose(self, node):
        return self.get_pose() == node.get_pose()

    def __hash__(self):
        return hash((self.pose,self.parent))

    def __eq__(self, node):
        return self.get_pose() == node.get_pose() and self.get_parent() == node.get_parent()

    def __repr__(self):
        return "Pose: {}".format(self.get_pose())