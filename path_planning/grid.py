import path_planning

class Grid:
    def __init__(self, occupancy):
        self.grid = occupancy
        self.rlim, self.clim = self.grid.shape
        self.neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1)
            ]

    def get_value(self, i, j):
        return self.grid[i, j]

    def get_shape(self):
        return self.grid.shape

    def get_adjacent(self, node):
        r,c = node.get_pose()
        adj = []
        for n in self.neighbors:
            new_r = r + n[0]
            new_c = c + n[1]
            if 0 <= new_r < self.rlim and 0 <= new_c < self.clim and not self.get_value(new_r, new_c):
                new_node = path_planning.Node()
                new_node.set_pose((new_r, new_c))
                new_node.set_parent(node)
                adj.append(new_node)
        return adj

    def is_open(self, pose):
        return not bool(self.get_value(*pose))

    def __repr__(self):
        return "{}".format(self.grid)