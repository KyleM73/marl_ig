import numpy as np

## Helper Functions
def nhood4(idx):
    out = np.zeros((4,2))
    ## Above cell
    out[0,0] = idx[0] - 1
    out[0,1] = idx[1]
    ## Left cell
    out[1,0] = idx[0]
    out[1,1] = idx[1] - 1
    ## Right cell
    out[2,0] = idx[0]
    out[2,1] = idx[1] + 1
    ## Below cell
    out[3,0] = idx[0] + 1
    out[3,1] = idx[1]

    return out

def nhood8(idx):
    out = np.zeros((8,2))
    ## Above Left cell
    out[0,0] = idx[0] - 1
    out[0,1] = idx[1] - 1
    ## Above cell
    out[1,0] = idx[0] - 1
    out[1,1] = idx[1]
    ## Above Right cell
    out[2,0] = idx[0] - 1
    out[2,1] = idx[1] + 1
    ## Left cell
    out[3,0] = idx[0]
    out[3,1] = idx[1] - 1
    ## Right cell
    out[4,0] = idx[0]
    out[4,1] = idx[1] + 1
    ## Below Left cell
    out[5,0] = idx[0] + 1
    out[5,1] = idx[1] - 1
    ## Below cell
    out[6,0] = idx[0] + 1
    out[6,1] = idx[1]
    ## Below Right cell
    out[7,0] = idx[0] + 1
    out[7,1] = idx[1] + 1

    return out

class FrontierExploration():
    def __init__(self, costmap, threshold):
        self.size_x = costmap.shape[0]
        self.size_y = costmap.shape[1]

        ## 0 values represent normal cells, 1 values represent frontier cells
        self.frontier_map = np.zeros((self.size_x,self.size_y), dtype=np.int8)

        self.completion = 0.0
        self.total_cells = self.size_x * self.size_y
        self.known_cells = 0
        for i in range(self.size_x):
            for j in range(self.size_y):
                if costmap[i,j] != 0.5:
                    self.known_cells += 1
        ## Maintain varible for map exploration percentage
        self.completion = float(self.known_cells / self.total_cells)

        ## Variable for exiting upon reaching threshold
        self.exploration_threshold = threshold

        ## Array of 2d coordinates containing cell idxs of frontier points
        self.frontier_points = np.zeros((1,2))

        print("Starting Percentage of Map Explored = ", self.completion*100.)

    def get_frontiers(self, costmap):
        # Part 1 - define all frontier cells as free points
        #          that have unknown neighbors
        
        ## use np.argwhere to get indices all free cells
        free_cells = np.argwhere(costmap == -1)

        frontier_indices = np.zeros((1,2),dtype = np.int8)
        first = True

        ## iterate through free cells
        for cell in free_cells:
            ## get neighbors
            neighbors = nhood4(cell)
            ## check if any neighbors are unknown
            for neighbor in neighbors:
                ## if neighbor is unknown
                if costmap[neighbor[0],neighbor[1]] == 0:
                    if first:
                        frontier_indices[0,0] = cell[0]
                        frontier_indices[0,1] = cell[1]
                        first = False
                        break ## only add cell once
                    else:
                        frontier_indices.push_back(cell)
                        break ## only add cell once

        ## Now we have labeled all free cells that have unknown 
        ##      neighbors as frontier points
        
        # Part 2 - generate up to 4 frontier centroids
        group = np.zeros((1,2),dtype=np.int8)
        group[0,0] = frontier_indices[0,0]
        group[0,1] = frontier_indices[0,1]
        for idx in frontier_indices:
            if ((idx[0] - group[0,0])**2 + (idx[1] - group[0,1])**2) < 50:
                group.push_back(idx)
        
        frontier_centroids = np.zeros((1,2))
        frontier_centroids = int(group.sum(axis=0)/group.shape[0])
        
        for idx in frontier_indices:
            if ((idx[0] - frontier_centroids[0,0])**2 + (idx[1] - frontier_centroids[0,1])**2) > 50:
                group = np.zeros((1,2),dtype=np.int8)
                group[0,0] = idx[0]
                group[0,1] = idx[1]
                break
        for idx in frontier_indices:
            if ((idx[0] - group[0,0])**2 + (idx[1] - group[0,1])**2) < 50:
                group.push_back(idx)
        tmp = int(group.sum(axis=0)/group.shape[0])
        frontier_centroids.push_back(tmp)

        for idx in frontier_indices:
            if ((idx[0] - frontier_centroids[0,0])**2 + (idx[1] - frontier_centroids[0,1])**2) > 50:
                group = np.zeros((1,2),dtype=np.int8)
                group[0,0] = idx[0]
                group[0,1] = idx[1]
                break
        for idx in frontier_indices:
            if ((idx[0] - group[0,0])**2 + (idx[1] - group[0,1])**2) < 50:
                group.push_back(idx)
        tmp = int(group.sum(axis=0)/group.shape[0])
        frontier_centroids.push_back(tmp)

        for idx in frontier_indices:
            if ((idx[0] - frontier_centroids[0,0])**2 + (idx[1] - frontier_centroids[0,1])**2) > 50:
                group = np.zeros((1,2),dtype=np.int8)
                group[0,0] = idx[0]
                group[0,1] = idx[1]
                break
        for idx in frontier_indices:
            if ((idx[0] - group[0,0])**2 + (idx[1] - group[0,1])**2) < 50:
                group.push_back(idx)
        tmp = int(group.sum(axis=0)/group.shape[0])
        frontier_centroids.push_back(tmp)

        return frontier_centroids
        
    def sample_frontiers(self, agent_pos):
        dist = (self.frontier_points[0,0] - agent_pos[0])**2 + (self.frontier_points[0,1] - agent_pos[1])**2
        waypoint = np.array([self.frontier_points[0,0],self.frontier_points[0,1]])

        for point in self.frontier_points:
            new_dist = (point[0] - agent_pos[0])**2 + (point[1] - agent_pos[1])**2
            if new_dist < dist:
                dist = new_dist
                waypoint = point
        
        return waypoint

    def get_next_waypoint(self, costmap, agent_pos):
        ## Update Exploration Completion
        for i in range(self.size_x):
            for j in range(self.size_y):
                if costmap[i,j] != 0.5:
                    self.known_cells += 1
        self.completion = float(self.known_cells / self.total_cells)
        
        ## Check Completion threshold
        if self.completion >= self.exploration_threshold:
            print("Exploration Task Completed")
            exit(69)
        else:
            print("Percentage of Map Explored = ", self.completion*100.)

        assert agent_pos[0] >= 0 and agent_pos[0] < self.size_x
        assert agent_pos[1] >= 0 and agent_pos[1] < self.size_y

        self.frontier_points = self.get_frontiers(costmap)
        waypoint = self.sample_frontiers(agent_pos)

        return waypoint