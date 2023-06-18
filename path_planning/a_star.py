import numpy as np

def A_Star(grid, start_node, target_node, max_iters=1000):
    if not grid.is_open(start_node.get_pose()):
        print("Invalid start pose.")
        print(start_node)
        return
    
    _open = []
    _closed = []
    _open.append(start_node)
    cnt = 0

    while _open and cnt < max_iters:
        cnt += 1
        min_f = np.argmin([n.get_f() for n in _open])
        current_node = _open.pop(min_f)
        if current_node.get_pose() in _closed:
            continue
        _closed.append(current_node.get_pose())
        if current_node.same_pose(target_node):
            break
        neighbors = grid.get_adjacent(current_node)
        for n in neighbors:
            if n.get_pose() in _closed:
                continue
            n.set_g(current_node.get_g() + 1)
            x1, y1 = n.get_pose()
            x2, y2 = target_node.get_pose()
            n.set_h((y2 - y1)**2 + (x2 - x1)**2)
            n.set_f(n.get_g() + n.get_h())
            _open.append(n)
    else:
        print("Solution not found.")
        return
    path = []
    while current_node.get_parent() is not None:
        path.append(current_node.get_pose())
        current_node = current_node.get_parent()
    path.append(current_node.get_pose())
    return path[::-1]
