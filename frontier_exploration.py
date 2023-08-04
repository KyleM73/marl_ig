import numpy as np

## Helper Functions
def nhood(idx,n=3):
    """
    arg: ndarray idx = [2,]
    arg: int n = 1,
    returns: ndarray neighbors = [n**2-1,2]
    returns indexes of the n**2-1 neighbors clipped to range [0,199]
    pattern:
    111
    101
    111
    """
    assert n>1 and n%2 #odd
    return np.clip(
        idx.reshape(1,2) + np.array(
        [[i,j] for i in range(-(n//2),n//2+1) for j in range(-(n//2),n//2+1) if i or j]
        ),0,199
    )

def map_nhood(idxs,func=nhood,**kwargs):
    """
    arg: ndarray = idxs [-1,2]
    arg: pyfunc func = ()
    returns : list(ndarray[]) = list of arrays containing neighbors of idx
    """
    return [func(idx,**kwargs) for idx in idxs]

def score_nhood(idx,costmap,func=nhood,**kwargs):
    """
    arg: ndarray idx = [2,]
    arg: ndarray costmap = [n,n]
    arg: pyfunc func = ()
    returns the sum of the absolute value of the nieghbors of a given idx
    idxs with mostly observed neighbors will return a higher score
    """
    neighbors = func(idx,**kwargs)
    return sum(np.abs(costmap[neighbors[:,0],neighbors[:,1]]))

def estimate_information_gain(frontier_candidates,costmap,n=3,normalize=True,**kwargs):
    """
    Docstring
    """
    info_gain = n**2 - 1 - np.vectorize(
            lambda r,c: score_nhood(np.array([r,c]),costmap,func=nhood,n=3,**kwargs)
            )(frontier_candidates[:,0],frontier_candidates[:,1])
    if normalize: info_gain /= (n**2 - 1)
    return info_gain

class FrontierExploration():
    def __init__(self,costmap,threshold):
        self.num_r,self.num_c = costmap.shape

        self.total_cells = self.num_r * self.num_c
        self.known_cells = np.sum(np.where(costmap!=0,1,0))
        self.completion = float(self.known_cells/self.total_cells) #percent of map explored
        self.exploration_threshold = threshold #percent of map to explore
        self.last_wp = np.zeros((2,2),dtype=np.int8)

        print("percentage of map explored: {}%".format(self.completion*100))

    def get_frontiers(self,costmap,agent_pose,num_candidates=15):
        """
        arg: ndarray costmap = [n,n]
        arg: ndarray agent_pose = [2]
        arg: int num_candidates = 1,
        return:
        finds frontier points to explore with conditions:
        1) min D1 distance to agent
        2) no neighbors that are obstacles
        3) must have some unobserved neighbors
        4) min D2 distance to other frontier points
        """
        min_dist_to_agent = 10
        min_dist_to_wall = 5
        min_dist_to_frontiers = 30
        
        free_cells = np.argwhere(costmap == -1) #cells observed to be empty
        #print("number of free cells: ",free_cells.shape[0])

        free_cells = free_cells[np.linalg.norm(free_cells-agent_pose.reshape(1,2),axis=1) > min_dist_to_agent] #`min_dist_to_agent` cells away from agent
        free_cells = free_cells[np.linalg.norm(free_cells,ord=np.inf,axis=1) < min(self.num_r,self.num_c)-min_dist_to_wall] #greater than `min_dist_to_wall` cells from edge
        free_cells = free_cells[np.linalg.norm(free_cells,ord=-np.inf,axis=1) > min_dist_to_wall] #greater than `min_dist_to_wall` cells from edge
        
        neighbors = map_nhood(free_cells,func=nhood,n=3)
        occupied_neighbor_mask = np.array([1 not in costmap[n[:,0],n[:,1]] for n in neighbors])
        free_cells = free_cells[occupied_neighbor_mask] #exclude cells adjacent to observbed objects
        neighbors = map_nhood(free_cells,func=nhood,n=3)
        unknown_neighbor_mask = np.array([not np.all(costmap[n[:,0],n[:,1]]==-1) for n in neighbors])
        free_cells = free_cells[unknown_neighbor_mask] #exclude cells with all observed neighbors
        #print("number of candidate frontier points: ",free_cells.shape[0])
        frontier_candidates = free_cells
    
        candidate_mask = np.zeros(frontier_candidates.shape[0]).astype(bool)
        #order candidates by information value - candidates with most unobserved neighbors first
        unobserved_neighbors = np.vectorize(
            lambda r,c: score_nhood(np.array([r,c]),costmap,func=nhood,n=3)
            )(frontier_candidates[:,0],frontier_candidates[:,1])
        ranked_neighbors = np.argsort(unobserved_neighbors) #fewest observed neighbors first
        frontier_candidates = frontier_candidates[ranked_neighbors] #reorder candidates
        candidate_mask[0] = True #pick the first candidate after ordering
        for i in range(1,frontier_candidates.shape[0]): #check all not yet selected candidates
            for j in range(frontier_candidates[candidate_mask].shape[0]): #check that target candidate `frontier_candidates[i]` is at least `min_dist_to_frontiers` from every other already selected frontier point
                if np.linalg.norm(frontier_candidates[i]-frontier_candidates[candidate_mask][j]) < min_dist_to_frontiers:
                    break
            else:
                candidate_mask[i] = True
            if frontier_candidates[candidate_mask].shape[0] == num_candidates:
                break
        #print("number of candidate frontier points: ",frontier_candidates[candidate_mask].shape[0])
        return frontier_candidates[candidate_mask]
    
    def sample_frontiers(self,frontier_candidates,costmap,agent_pose,agent_num):
        """
        docstring
        cost = -travel_dist + dist_from_other_agents_last_wp (antidist)
        """
        min_dist_to_last_wp = 5
        c0,c1,c2 = 10,-1,-1
        n = 7
        max_dist = (max(self.num_r,self.num_c)*np.sqrt(2))

        frontier_candidates = frontier_candidates[np.linalg.norm(self.last_wp[agent_num]-frontier_candidates,axis=1) > min_dist_to_last_wp] #exclude candidates too close to last selected waypoint
        
        heuristic_travel_dist = np.linalg.norm(frontier_candidates-agent_pose,axis=1)
        dist_from_opposing_agent_last_wp = np.linalg.norm(frontier_candidates-self.last_wp[int(not agent_num)],axis=1)
        info_gain_est = estimate_information_gain(frontier_candidates,costmap,n,normalize=True)
        return frontier_candidates[np.argmin(
              c0*heuristic_travel_dist/max_dist
            + c1*dist_from_opposing_agent_last_wp/max_dist
            + c2*info_gain_est
            )]
    
        # try: get distance between all candidates
        # pick candidate with most close by neighbors
        # with highest info gain

    def get_next_waypoint(self,costmap,agent_pose,agent_num):
        """
        docstring
        """
        agent_pose = np.array(agent_pose)
        self.known_cells = np.sum(np.where(costmap!=0,1,0))
        self.completion = float(self.known_cells/self.total_cells) #update completion percent
        
        if self.completion >= self.exploration_threshold: #check completion threshold
            print("exploration complete")
            return np.array([[0,0]])
        else:
            print("percentage of map explored: {}%".format(self.completion*100))

        frontier_candidates = self.get_frontiers(costmap,agent_pose)
        waypoint = self.sample_frontiers(frontier_candidates,costmap,agent_pose,agent_num)

        self.last_wp[agent_num] = waypoint

        print("agent {} position =".format(agent_num),agent_pose)
        print("agent {} waypoint =".format(agent_num),waypoint)

        return waypoint