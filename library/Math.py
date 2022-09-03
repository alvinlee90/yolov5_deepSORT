import numpy as np

# Option to rotate and shift (for label info)
def create_corners(dimension, location=None, R=None):
    dx = dimension[2] / 2
    dy = dimension[0] / 2
    dz = dimension[1] / 2

    x_corners = []
    y_corners = []
    z_corners = []

    for i in [1, -1]:
        for j in [1,-1]:
            for k in [1,-1]:
                x_corners.append(dx*i)
                y_corners.append(dy*j)
                z_corners.append(dz*k)

    corners = [x_corners, y_corners, z_corners]

    # rotate if R is passed in
    if R is not None:
        corners = np.dot(R, corners)

    # shift if location is passed in
    if location is not None:
        for i,loc in enumerate(location):
            corners[i,:] = corners[i,:] + loc

    final_corners = []
    for i in range(8):
        final_corners.append([corners[0][i], corners[1][i], corners[2][i]])

    return final_corners


def calc_location(detected_object, alpha):
    # Global orientation
    R = rot_y(detected_object.rotation_y)

    # 2D bounding box - left top right bottom
    box_corners = [detected_object.xmin, detected_object.ymin, \
                    detected_object.xmax, detected_object.ymax]

    # get the point constraints
    constraints = []

    left_constraints = []
    right_constraints = []
    top_constraints = []
    bottom_constraints = []

    # using a different coord system
    dx = detected_object.lenght / 2
    dy = detected_object.height / 2
    dz = detected_object.width / 2

    # Based on the relative angle, a different configuration occurs
    # negative is back of car, positive is front
    # Straight but opposite way (away)
    if alpha < np.deg2rad(92) and alpha > np.deg2rad(88):
        left_mult = 1
        right_mult = 1
    # Straight but same way (towards)
    elif alpha < np.deg2rad(-88) and alpha > np.deg2rad(-92):
        left_mult = -1
        right_mult = -1
    # Prependicular and towards the right
    elif alpha < np.deg2rad(90) and alpha > -np.deg2rad(90):
        left_mult = -1
        right_mult = 1
    # Prependicular and towards the left
    else:
        left_mult = 1
        right_mult = -1

    # if the car is facing the oppositeway, switch left and right
    if alpha > 0:
        switch_mult = 1
    else: 
        switch_mult = -1

    # left and right could either be the front of the car ot the back of the car
    # careful to use left and right based on image, no of actual car's left and right
    for i in (-1,1):
        left_constraints.append([left_mult * dx, i*dy, -switch_mult * dz])
    for i in (-1,1):
        right_constraints.append([right_mult * dx, i*dy, switch_mult * dz])

    # top and bottom are easy, just the top and bottom of car
    for i in (-1,1):
        for j in (-1,1):
            top_constraints.append([i*dx, -dy, j*dz])
    for i in (-1,1):
        for j in (-1,1):
            bottom_constraints.append([i*dx, dy, j*dz])

    # now, 64 combinations
    for left in left_constraints:
        for top in top_constraints:
            for right in right_constraints:
                for bottom in bottom_constraints:
                    constraints.append([left, top, right, bottom])

    # filter out the ones with repeats
    constraints = filter(lambda x: len(x) == len(set(tuple(i) for i in x)), constraints)

    best_loc = None
    best_error = [1e09]

    # loop through each possible constraint, hold on to the best guess
    # constraint will be 64 sets of 4 corners
    for constraint in constraints:
        # each corner
        Xa = constraint[0]  # Left constraint
        Xb = constraint[1]  # Top constraint
        Xc = constraint[2]  # Right constraint
        Xd = constraint[3]  # Bottom constraint

        X_array = [Xa, Xb, Xc, Xd]

        # M: all 1's down diagonal, and upper 3x1 is Rotation_matrix * [x, y, z]
        Ma = np.identity(4)
        Mb = np.identity(4)
        Mc = np.identity(4)
        Md = np.identity(4)

        M_array = [Ma, Mb, Mc, Md]

        # create A, b
        A = np.zeros([4,3], dtype=np.float)
        b = np.zeros([4,1])

        indicies = [0, 1, 0, 1] # [xmin, ymin, xmax, ymax]
        for row, index in enumerate(indicies):
            X = X_array[row]        # Constraint 
            M = M_array[row]

            RX = np.dot(R, X)       # Rotate the constraint 
            M[:3,3] = RX.reshape(3)

            M = np.dot(detected_object.proj_matrix, M) # Project constraint to image plane

            A[row, :] = M[index,:3] - box_corners[row] * M[2,:3]
            b[row] = box_corners[row] * M[2,3] - M[index,3]

        # solve here with least squares, since over fit will get some error
        loc, error, rank, s = np.linalg.lstsq(A, b, rcond=None)

        # Check estimation 
        if error < best_error:
            best_loc = loc
            best_error = error

    return [best_loc[0][0], best_loc[1][0] + dy, best_loc[2][0]]

def rot_y(x):
    ''' Homogenous rectifying rotation matrix about the y-axis. '''
    c = np.cos(x)
    s = np.sin(x)

    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]]).reshape([3,3])
