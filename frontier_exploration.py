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

    for i in range(4):
        out[i,0] = np.clip(out[i,0],0,199)
        out[i,1] = np.clip(out[i,1],0,199)

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
                if costmap[i,j] != 0:
                    self.known_cells += 1
        ## Maintain varible for map exploration percentage
        self.completion = float(self.known_cells / self.total_cells)

        ## Variable for exiting upon reaching threshold
        self.exploration_threshold = threshold

        ## Array of 2d coordinates containing cell idxs of frontier points
        self.frontier_points = np.zeros((1,2))
        self.prev_wp = np.zeros((1,2),dtype = np.int8)

        print("Starting Percentage of Map Explored = ", self.completion*100.)

    def get_frontiers(self, costmap):
        # Part 1 - define all frontier cells as free points
        #          that have unknown neighbors
        
        ## use np.argwhere to get indices all free cells
        free_cells = np.argwhere(costmap == -1)
        # print("number of frontier candidates = ",free_cells.shape)

        frontier_indices = np.zeros((1,2),dtype = np.int8)
        tmp_cell = np.zeros((1,2),dtype = np.int8)
        first = True

        ## iterate through free cells
        for cell in free_cells:
            ## get 4 neighbors
            neighbors = nhood4(cell)

            ## check if any neighbors are unknown
            for neighbor in neighbors:
                # print("neighbor.shape = ",neighbor.shape)

                ## Ignore points with occupied neighbors
                if costmap[int(neighbor[0]),int(neighbor[1])] == 1:
                    continue

                ## if neighbor is unknown
                if costmap[int(neighbor[0]),int(neighbor[1])] == 0:
                    if first:
                        frontier_indices[0,0] = cell[0]
                        frontier_indices[0,1] = cell[1]
                        first = False
                        break ## only add cell once
                    else:
                        # print("frontier_indices.shape = ",frontier_indices.shape)
                        # print("cell.shape = ",cell.shape)
                        tmp_cell[0,0] = cell[0]
                        tmp_cell[0,1] = cell[1]
                        frontier_indices = np.append(frontier_indices,tmp_cell,axis=0)
                        break ## only add cell once

        ## Now we have labeled all free cells that have unknown 
        ##      neighbors as frontier points

        # print("frontier_indices = ", frontier_indices)
        # print("frontier_indices.shape = ", frontier_indices.shape)
        # print("frontier_indices.dtype = ", frontier_indices.dtype)
        
        # Part 2 - generate up to 4 frontier centroids (frontier centroids are sample points)  
        frontier_centroids = np.zeros((1,2),dtype=np.int8)
        tmp_idx = np.zeros((1,2),dtype=np.int8)
        ## Consider the first frontier index as a frontier cluser
        for idx in frontier_indices:
            frontier_centroids[0,0] = int(idx[0])
            frontier_centroids[0,1] = int(idx[1])
            break
        # print("frontier_centroids.shape = ", frontier_centroids.shape)
        ## Ignore first point
        i = 1
        for idx in frontier_indices[1:]:
            ## Group next frontier points outside 5 meters from original, not at map bounds
            if np.sqrt((idx[0] - frontier_centroids[0,0])**2 + (idx[1] - frontier_centroids[0,1])**2) > 30. and \
                idx[0] > 5 and idx[0] < 194 and idx[1] > 5 and idx[1] < 194:
                tmp_idx[0,0] = idx[0]
                tmp_idx[0,1] = idx[1]
                frontier_centroids = np.append(frontier_centroids, tmp_idx,axis=0)
                index_added = i
                break
            i += 1

        # print("index added 2nd = ", index_added)
        # print("frontier_centroids.shape = ", frontier_centroids.shape)

        tmp_idx = np.zeros((1,2),dtype=np.int8)
        j = index_added
        if(frontier_centroids.shape[0]>=2):
            ## Ignore all points already considered
            for idx in frontier_indices[index_added:]:
                ## Group next frontier points outside 5 meters from original, not at map bounds
                if np.sqrt((idx[0] - frontier_centroids[0,0])**2 + (idx[1] - frontier_centroids[0,1])**2) > 30. and \
                np.sqrt((idx[0] - frontier_centroids[1,0])**2 + (idx[1] - frontier_centroids[1,1])**2) > 30. and \
                    idx[0] > 5 and idx[0] < 194 and idx[1] > 5 and idx[1] < 194:
                    tmp_idx[0,0] = idx[0]
                    tmp_idx[0,1] = idx[1]
                    frontier_centroids = np.append(frontier_centroids, tmp_idx,axis=0)
                    index_added = j
                    break
                j += 1
            # print("index added 3rd = ", index_added)
            # print("frontier_centroids.shape = ", frontier_centroids.shape)

        tmp_idx = np.zeros((1,2),dtype=np.int8)
        k = index_added
        if(frontier_centroids.shape[0]>=3):
            ## Ignore all points already considered
            for idx in frontier_indices[index_added:]:
                ## Group next frontier points outside 5 meters from original, not at map bounds
                if np.sqrt((idx[0] - frontier_centroids[0,0])**2 + (idx[1] - frontier_centroids[0,1])**2) > 30. and \
                np.sqrt((idx[0] - frontier_centroids[1,0])**2 + (idx[1] - frontier_centroids[1,1])**2) > 30. and \
                np.sqrt((idx[0] - frontier_centroids[2,0])**2 + (idx[1] - frontier_centroids[2,1])**2) > 30. and \
                    idx[0] > 5 and idx[0] < 194 and idx[1] > 5 and idx[1] < 194:
                    tmp_idx[0,0] = idx[0]
                    tmp_idx[0,1] = idx[1]
                    index_added = k
                    frontier_centroids = np.append(frontier_centroids, tmp_idx,axis=0)
                    break
                k += 1
            # print("index added 4th = ", index_added)
            # print("frontier_centroids.shape = ", frontier_centroids.shape)

        # print("frontier_centroids.shape = ",frontier_centroids.shape)
        print("frontier_centroids = ",frontier_centroids)

        return frontier_centroids
        
    def sample_frontiers(self, agent_pos, agent_num):
        # print("self.frontier_points.shape[0] = ",self.frontier_points.shape)
        ## Send dummies in case logic below fails
        if agent_num == 0:
            waypoint = np.array([110,90])
        elif agent_num == 1:
            waypoint = np.array([90,110])
        dist = 0.

        ## Mechanism to ensure we don't repeat wps to agents
        for j in range(self.frontier_points.shape[0]):
            # print("self.frontier_points.shape = ",self.frontier_points.shape)
            # print("self.prev_wp.shape = ",self.prev_wp.shape)
            if self.prev_wp[0,0] == self.frontier_points[j,0] and self.prev_wp[0,1] == self.frontier_points[j,1]:
                self.frontier_points = np.delete(self.frontier_points, j, axis=0)
                break

        ## Get the first wp in the list that meets bounds criteria
        for i in range(self.frontier_points.shape[0]):
            if self.frontier_points[0,0] > 5 and self.frontier_points[0,0] < 194 and self.frontier_points[0,1] > 5 and self.frontier_points[0,1] < 194:
                waypoint = np.array([self.frontier_points[i,0],self.frontier_points[i,1]])
                dist = (self.frontier_points[i,0] - agent_pos[0])**2 + (self.frontier_points[i,1] - agent_pos[1])**2
                selected_index = i
                break

        ## Iterate through remaining wps in list 
        for point in self.frontier_points[selected_index:]:
            new_dist = (point[0] - agent_pos[0])**2 + (point[1] - agent_pos[1])**2
            if new_dist < dist and point[0] > 5 and point[0] < 194 and point[1] > 5 and point[1] < 194:
                dist = new_dist
                waypoint = point
        
        ## Only will avoid repeated wps in dual agent case
        if agent_num == 0:
            self.prev_wp[0,0] = waypoint[0]
            self.prev_wp[0,1] = waypoint[1]
        print("waypoint = ", tuple(waypoint))
        return tuple(waypoint)

    def get_next_waypoint(self, costmap, agent_pos, agent_num):

        print("getting next waypoint for agent ",agent_num)

        # print("costmap = ",costmap)

        ## Update Exploration Completion
        self.known_cells = 0
        for i in range(self.size_x):
            for j in range(self.size_y):
                if costmap[i,j] != 0:
                    self.known_cells += 1
        self.completion = float(self.known_cells / self.total_cells)
        
        ## Check Completion threshold
        if self.completion >= self.exploration_threshold:
            print("Exploration Task Completed")
            # exit(69)
        else:
            print("Percentage of Map Explored = ", self.completion*100.)

        assert agent_pos[0][0] >= 0 and agent_pos[0][0] < self.size_x
        assert agent_pos[0][1] >= 0 and agent_pos[0][1] < self.size_y

        position = np.zeros(2)
        position[0] = agent_pos[0][0]
        position[1] = agent_pos[0][1]
        # print("agent position = ",position)

        self.frontier_points = self.get_frontiers(costmap)
        waypoint = self.sample_frontiers(position,agent_num)

        return waypoint