from math import sqrt
import numpy as np

def euclidean_distance(*coords):
    """
    Examples
        2D euclidean distance
            point_1 = (x1, y1)
            point_2 = (x2, y2)
            distance = sqrt((x2 - x1)**2 + (y2 - y1)**2)
        3D euclidean distance
            point_1 = (x1, y1, z1)
            point_2 = (x2, y2, z2)
            distance = sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    """
    coords_matrix = np.array([coords])
    
    # square_sum = 0
    # for axis in coords_matrix.T:
    #     square = (axis[1] - axis[0])**2
    #     square_sum += square

    square_sum = sum([(axis[1] - axis[0])**2 for axis in coords_matrix.T])
    distance = sqrt(square_sum)
    return distance

def minkowski_distance():
    pass
