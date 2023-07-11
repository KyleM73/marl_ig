import numpy as np
import copy

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

def nhood16(idx):
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
    ########
    ## Above Left cell
    out[0,0] = idx[0] - 2
    out[0,1] = idx[1] - 2
    ## Above cell
    out[1,0] = idx[0] - 2
    out[1,1] = idx[1]
    ## Above Right cell
    out[2,0] = idx[0] - 2
    out[2,1] = idx[1] + 2
    ## Left cell
    out[3,0] = idx[0]
    out[3,1] = idx[1] - 2
    ## Right cell
    out[4,0] = idx[0]
    out[4,1] = idx[1] + 2
    ## Below Left cell
    out[5,0] = idx[0] + 2
    out[5,1] = idx[1] - 2
    ## Below cell
    out[6,0] = idx[0] + 2
    out[6,1] = idx[1]
    ## Below Right cell
    out[7,0] = idx[0] + 2
    out[7,1] = idx[1] + 2

    return out


# def nhood16(idx):
#     out = np.zeros((8,2))
#     ## Above cell
#     out[0,0] = idx[0] - 1
#     out[0,1] = idx[1]
#     ## Left cell
#     out[1,0] = idx[0]
#     out[1,1] = idx[1] - 1
#     ## Right cell
#     out[2,0] = idx[0]
#     out[2,1] = idx[1] + 1
#     ## Below cell
#     out[3,0] = idx[0] + 1
#     out[3,1] = idx[1]
#     ## Above cell
#     out[0,0] = idx[0] - 2
#     out[0,1] = idx[1]
#     ## Left cell
#     out[1,0] = idx[0]
#     out[1,1] = idx[1] - 2
#     ## Right cell
#     out[2,0] = idx[0]
#     out[2,1] = idx[1] + 2
#     ## Below cell
#     out[3,0] = idx[0] + 2
#     out[3,1] = idx[1]
#     ## Above cell
#     out[0,0] = idx[0] - 3
#     out[0,1] = idx[1]
#     ## Left cell
#     out[1,0] = idx[0]
#     out[1,1] = idx[1] - 3
#     ## Right cell
#     out[2,0] = idx[0]
#     out[2,1] = idx[1] + 3
#     ## Below cell
#     out[3,0] = idx[0] + 3
#     out[3,1] = idx[1]
#     ## Above cell
#     out[0,0] = idx[0] - 4
#     out[0,1] = idx[1]
#     ## Left cell
#     out[1,0] = idx[0]
#     out[1,1] = idx[1] - 4
#     ## Right cell
#     out[2,0] = idx[0]
#     out[2,1] = idx[1] + 4
#     ## Below cell
#     out[3,0] = idx[0] + 4
#     out[3,1] = idx[1]

    for i in range(8):
        out[i,0] = np.clip(out[i,0],0,199)
        out[i,1] = np.clip(out[i,1],0,199)

    return out

## TODO: 
# 3) now that we have a better list, filter to leave only those at least so far from each other
# 4) last thing is to select the waypoints based on something besides only distance

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
        self.candidate_waypoints = np.zeros((1,2))
        self.prev_wps = np.zeros((2,2),dtype=np.int8)
        self.agent0_wp = np.zeros((1,2),dtype = np.int8)

        self.agent0_pos = np.zeros(2)
        self.agent1_pos = np.zeros(2)

        print("Starting Percentage of Map Explored = ", self.completion*100.)

    def get_frontiers(self, costmap, agent_pos):
        # Part 1 - define all frontier cells as free points
        #          that have unknown neighbors
        
        ## get indices all free cells
        free_cells = np.argwhere(costmap == -1)
        # print("number of free cells = ",free_cells.shape)

        count_free = -1
        remove_rows = []

        ## Remove indices near agent
        for cell in free_cells:
            count_free += 1
            if np.sqrt( (cell[0] - agent_pos[0])**2 + (cell[1]- agent_pos[1])**2 ) < 10:
                remove_rows.append(count_free)
        free_cells = np.delete(free_cells,remove_rows,axis=0)


        count_free = -1
        remove_rows = []
        ## Remove indices near edges
        for cell in free_cells:
            count_free += 1
            if cell[0] >= 194 or cell[0] <= 5:
                remove_rows.append(count_free)
                continue
            if cell[1] >= 194 or cell[1] <= 5:
                remove_rows.append(count_free)
                continue
        # print("rows to be removed = ",remove_rows)
        free_cells = np.delete(free_cells,remove_rows,axis=0)
        # print("number of free cells after edge removal = ",free_cells.shape)

        count_free = -1
        remove_rows = []
        # print("number of free cells = ",free_cells.shape)

        ## Remove indices with occupied neighbors
        for cell in free_cells:
            count_free += 1
            neighbors = nhood4(cell)
            ## Remove cells with occupied neighbors
            for neighbor in neighbors:
                if costmap[int(neighbor[0]),int(neighbor[1])] == 1:
                    remove_rows.append(count_free)
                    break
        # print("rows to be removed = ",remove_rows)
        free_cells = np.delete(free_cells,remove_rows,axis=0)
        # print("number of free cells after occ neighbor removal = ",free_cells.shape)

        count_free = -1
        remove_rows = []

        # print("number of free cells = ",free_cells.shape)
        ## Remove indices with all known neighbors
        for cell in free_cells:
            free_neighbor_count = 0
            count_free += 1
            neighbors = nhood4(cell)
            for neighbor in neighbors:
                if costmap[int(neighbor[0]),int(neighbor[1])] == -1:
                    free_neighbor_count += 1
            if free_neighbor_count >= 4:
                remove_rows.append(count_free)
        # print("rows to be removed = ",remove_rows)
        frontier_indices = np.delete(free_cells,remove_rows,axis=0)
        # print("number of free cells after free neighbor removal = ",free_cells.shape)        

        ## Now we have labeled all free cells that 0) are too close to agent
        ## 1) are not near edges; 2) have occupied neighbor cells
        ## or 3) have only known neighbor cells

        ## Now remove points that are near by each other
        ## Below we generate up to 15 candidate frontiers, 
        ## all more than 30 cells distance from each other

        tmp_idx = frontier_indices[0,:]
        num_pts = 0
        count_free = 0 # here we start at index 1 in the loop
        remove_rows = []
        ## Remove points near point 0
        for cell in frontier_indices[1:]:
            count_free += 1
            dist = np.sqrt(pow(cell[0] - tmp_idx[0],2) + pow(cell[1] - tmp_idx[1],2))
            if dist <= 30.:
                remove_rows.append(count_free)

        frontier_indices = np.delete(frontier_indices,remove_rows,axis=0)

        tmp_idx = frontier_indices[1,:]
        num_pts += 1
        count_free = 1 # here we start at index 2 in the loop
        remove_rows = []
        ## Remove points near point 1
        for cell in frontier_indices[2:]:
            count_free += 1
            dist = np.sqrt(pow(cell[0] - tmp_idx[0],2) + pow(cell[1] - tmp_idx[1],2))
            if dist <= 30.:
                remove_rows.append(count_free)
        frontier_indices = np.delete(frontier_indices,remove_rows,axis=0)

        if frontier_indices.shape[0] > 3:
            tmp_idx = frontier_indices[2,:] 
            num_pts += 1
            count_free = 2 # here we start at index 3 in the loop
            remove_rows = []
            ## Remove points near point 2
            for cell in frontier_indices[3:]:
                count_free += 1
                dist = np.sqrt(pow(cell[0] - tmp_idx[0],2) + pow(cell[1] - tmp_idx[1],2))
                if dist <= 40.:
                    remove_rows.append(count_free)
            frontier_indices = np.delete(frontier_indices,remove_rows,axis=0)

        if frontier_indices.shape[0] > 4:
            tmp_idx = frontier_indices[3,:] 
            num_pts += 1
            count_free = 3 # here we start at index 4 in the loop
            remove_rows = []
            ## Remove points near point 2
            for cell in frontier_indices[4:]:
                count_free += 1
                dist = np.sqrt(pow(cell[0] - tmp_idx[0],2) + pow(cell[1] - tmp_idx[1],2))
                if dist <= 60.:
                    remove_rows.append(count_free)
            frontier_indices = np.delete(frontier_indices,remove_rows,axis=0)

        if frontier_indices.shape[0] > 5:
            tmp_idx = frontier_indices[4,:] 
            num_pts += 1
            count_free = 4 # here we start at index 5 in the loop
            remove_rows = []
            ## Remove points near point 2
            for cell in frontier_indices[5:]:
                count_free += 1
                dist = np.sqrt(pow(cell[0] - tmp_idx[0],2) + pow(cell[1] - tmp_idx[1],2))
                if dist <= 60.:
                    remove_rows.append(count_free)
            frontier_indices = np.delete(frontier_indices,remove_rows,axis=0)

        if frontier_indices.shape[0] > 6:
            tmp_idx = frontier_indices[5,:] 
            num_pts += 1
            count_free = 5 # here we start at index 5 in the loop
            remove_rows = []
            ## Remove points near point 2
            for cell in frontier_indices[6:]:
                count_free += 1
                dist = np.sqrt(pow(cell[0] - tmp_idx[0],2) + pow(cell[1] - tmp_idx[1],2))
                if dist <= 60.:
                    remove_rows.append(count_free)
            frontier_indices = np.delete(frontier_indices,remove_rows,axis=0)

        if frontier_indices.shape[0] > 7:
            tmp_idx = frontier_indices[6,:] 
            num_pts += 1
            count_free = 6 # here we start at index 5 in the loop
            remove_rows = []
            ## Remove points near point 2
            for cell in frontier_indices[7:]:
                count_free += 1
                dist = np.sqrt(pow(cell[0] - tmp_idx[0],2) + pow(cell[1] - tmp_idx[1],2))
                if dist <= 60.:
                    remove_rows.append(count_free)
            frontier_indices = np.delete(frontier_indices,remove_rows,axis=0)

        if frontier_indices.shape[0] > 8:
            tmp_idx = frontier_indices[7,:] 
            num_pts += 1
            count_free = 7 # here we start at index 5 in the loop
            remove_rows = []
            ## Remove points near point 2
            for cell in frontier_indices[8:]:
                count_free += 1
                dist = np.sqrt(pow(cell[0] - tmp_idx[0],2) + pow(cell[1] - tmp_idx[1],2))
                if dist <= 60.:
                    remove_rows.append(count_free)
            frontier_indices = np.delete(frontier_indices,remove_rows,axis=0)
        
        # print("frontier_indices.shape[0] = ",frontier_indices.shape)

        if frontier_indices.shape[0] > 9:
            tmp_idx = frontier_indices[8,:] 
            num_pts += 1
            count_free = 8 # here we start at index 5 in the loop
            remove_rows = []
            ## Remove points near point 2
            for cell in frontier_indices[9:]:
                count_free += 1
                dist = np.sqrt(pow(cell[0] - tmp_idx[0],2) + pow(cell[1] - tmp_idx[1],2))
                if dist <= 60.:
                    remove_rows.append(count_free)
            frontier_indices = np.delete(frontier_indices,remove_rows,axis=0)
        
        # print("frontier_indices.shape[0] = ",frontier_indices.shape)

        if frontier_indices.shape[0] > 10:
            tmp_idx = frontier_indices[9,:] 
            num_pts += 1
            count_free = 9 # here we start at index 5 in the loop
            remove_rows = []
            ## Remove points near point 2
            for cell in frontier_indices[10:]:
                count_free += 1
                dist = np.sqrt(pow(cell[0] - tmp_idx[0],2) + pow(cell[1] - tmp_idx[1],2))
                if dist <= 60.:
                    remove_rows.append(count_free)
            frontier_indices = np.delete(frontier_indices,remove_rows,axis=0)


        if frontier_indices.shape[0] > 11:
            tmp_idx = frontier_indices[10,:] 
            num_pts += 1
            count_free = 10 # here we start at index 5 in the loop
            remove_rows = []
            ## Remove points near point 2
            for cell in frontier_indices[11:]:
                count_free += 1
                dist = np.sqrt(pow(cell[0] - tmp_idx[0],2) + pow(cell[1] - tmp_idx[1],2))
                if dist <= 60.:
                    remove_rows.append(count_free)
            frontier_indices = np.delete(frontier_indices,remove_rows,axis=0)

        if frontier_indices.shape[0] > 12:
            tmp_idx = frontier_indices[11,:] 
            num_pts += 1
            count_free = 11 # here we start at index 5 in the loop
            remove_rows = []
            ## Remove points near point 2
            for cell in frontier_indices[12:]:
                count_free += 1
                dist = np.sqrt(pow(cell[0] - tmp_idx[0],2) + pow(cell[1] - tmp_idx[1],2))
                if dist <= 60.:
                    remove_rows.append(count_free)
            frontier_indices = np.delete(frontier_indices,remove_rows,axis=0)

        if frontier_indices.shape[0] > 13:
            tmp_idx = frontier_indices[12,:] 
            num_pts += 1
            count_free = 12 # here we start at index 5 in the loop
            remove_rows = []
            ## Remove points near point 2
            for cell in frontier_indices[13:]:
                count_free += 1
                dist = np.sqrt(pow(cell[0] - tmp_idx[0],2) + pow(cell[1] - tmp_idx[1],2))
                if dist <= 60.:
                    remove_rows.append(count_free)
            frontier_indices = np.delete(frontier_indices,remove_rows,axis=0)


        if frontier_indices.shape[0] > 14:
            tmp_idx = frontier_indices[13,:] 
            num_pts += 1
            count_free = 13 # here we start at index 5 in the loop
            remove_rows = []
            ## Remove points near point 2
            for cell in frontier_indices[14:]:
                count_free += 1
                dist = np.sqrt(pow(cell[0] - tmp_idx[0],2) + pow(cell[1] - tmp_idx[1],2))
                if dist <= 60.:
                    remove_rows.append(count_free)
            frontier_indices = np.delete(frontier_indices,remove_rows,axis=0)


        if frontier_indices.shape[0] > 15:
            tmp_idx = frontier_indices[14,:] 
            num_pts += 1
            count_free = 14 # here we start at index 5 in the loop
            remove_rows = []
            ## Remove points near point 2
            for cell in frontier_indices[15:]:
                count_free += 1
                dist = np.sqrt(pow(cell[0] - tmp_idx[0],2) + pow(cell[1] - tmp_idx[1],2))
                if dist <= 60.:
                    remove_rows.append(count_free)
            frontier_indices = np.delete(frontier_indices,remove_rows,axis=0)
        
        print("length of frontiers before removing extras not considered = ",frontier_indices.shape[0])
        frontier_indices = frontier_indices[0:num_pts]  
        print("Number of candidate frontiers = ",frontier_indices.shape[0])

        return frontier_indices
        
    
    ## Now that we have each other agent pos, what we can do is do a cost = np.zeros((2,len(self.candidate_waypoints)))
    ## and include the score of dist from self and dist from other agent; first equally weighted
    def sample_frontiers(self, agent_pos, agent_num, costmap):

        ## remove candidates from the list if the previous agent is already assigned them
        if agent_num == 1:
            for j in range(self.candidate_waypoints.shape[0]):
                if abs(self.prev_wps[0,0] - self.candidate_waypoints[j,0]) + abs(self.prev_wps[0,1] - self.candidate_waypoints[j,1]) <= 5:
                    self.candidate_waypoints = np.delete(self.candidate_waypoints, j, axis=0)
                    print("Agent 0 waypoint removed")
                    break

        
        cost = np.zeros(self.candidate_waypoints.shape[0])
        ## cost = -travel_dist + dist_from_other_agents_prev_wp (antidist)
        for i in range(cost.shape[0]):
            # travel dist
            dist = np.sqrt(pow(self.candidate_waypoints[i,0] - agent_pos[0],2) + pow(self.candidate_waypoints[i,1] - agent_pos[1],2))

            # dist from other agent last wp
            if not agent_num: # i.e. agent 0
                antidist = np.sqrt(pow(self.candidate_waypoints[i,0] - self.prev_wps[1,0],2) + pow(self.candidate_waypoints[i,1] - self.prev_wps[1,1],2))
            else:
                antidist = np.sqrt(pow(self.candidate_waypoints[i,0] - self.prev_wps[0,0],2) + pow(self.candidate_waypoints[i,1] - self.prev_wps[0,1],2))

            # info gain
            ## First we find number of neighbors unknown at the candidate point
            cell = self.candidate_waypoints[i,:]
            neighbors = nhood16(cell)
            count = 0
            for neighbor in neighbors:
                # print("neighbor.shape = ",neighbor.shape)
                j = int(neighbor[0])
                k = int(neighbor[1])
                # print("i,j = ", i, ", ",j)
                if costmap[j,k] == 0:
                    count += 1
            ## Next we find number of neighbors at points along the path
            midway = np.zeros(2,dtype = np.int8)
            midway[0] = int( (agent_pos[0] + cell[0]) / 4)
            midway[1] = int( (agent_pos[1] + cell[1]) / 4)
            neighbors = nhood16(midway)
            for neighbor in neighbors:
                # print("neighbor.shape = ",neighbor.shape)
                j = int(neighbor[0])
                k = int(neighbor[1])
                # print("i,j = ", i, ", ",j)
                if costmap[j,k] == 0:
                    count += 1

            midway = np.zeros(2,dtype = np.int8)
            midway[0] = int( 2 * (agent_pos[0] + cell[0]) / 4)
            midway[1] = int( 2 * (agent_pos[1] + cell[1]) / 4)
            neighbors = nhood16(midway)
            for neighbor in neighbors:
                # print("neighbor.shape = ",neighbor.shape)
                j = int(neighbor[0])
                k = int(neighbor[1])
                # print("i,j = ", i, ", ",j)
                if costmap[j,k] == 0:
                    count += 1

            midway = np.zeros(2,dtype = np.int8)
            midway[0] = int( 3 * (agent_pos[0] + cell[0]) / 3)
            midway[1] = int( 3 * (agent_pos[1] + cell[1]) / 3)
            neighbors = nhood16(midway)
            for neighbor in neighbors:
                # print("neighbor.shape = ",neighbor.shape)
                j = int(neighbor[0])
                k = int(neighbor[1])
                # print("i,j = ", i, ", ",j)
                if costmap[j,k] == 0:
                    count += 1

            print("number of unknown neighbors + those of midway point = ", count)
            cost[i] = (100) * count# / dist / 10 # + (-0.1) * dist + (2) * antidist

        print("cost = ",cost)
        opt_idx = np.argmax(cost)

        if agent_num == 1:
            self.prev_wps[1,:] = self.candidate_waypoints[opt_idx,:]
        else:
            self.prev_wps[0,:] = self.candidate_waypoints[opt_idx,:]

        return tuple(self.candidate_waypoints[opt_idx,:])


    def get_next_waypoint(self, costmap, agent_pos, agent_num):

        print("getting next waypoint for agent ",agent_num)
        if agent_num == 1:
            self.agent1_pos[0] = agent_pos[0][0]
            self.agent1_pos[0] = agent_pos[0][1]
        else:
            self.agent0_pos[0] = agent_pos[0][0]
            self.agent0_pos[0] = agent_pos[0][1]

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

        position = np.zeros(2)
        position[0] = agent_pos[0][0]
        position[1] = agent_pos[0][1]
        # print("agent position = ",position)

        self.candidate_waypoints = self.get_frontiers(costmap, position)
        waypoint = self.sample_frontiers(position,agent_num,costmap)

        print("agent {} position =".format(agent_num),agent_pos)
        print("agent {} waypoint =".format(agent_num),waypoint)

        return waypoint