def contourRadius(contour):
    """
    calculates the radius of a list of a (x,y) points

    args:
    	@a contour, numpy array (num points, 2)
    """
    if len(contour) < 3:
        return 0.0
    tup = zip(contour[:,0],contour[:,1])

    p = Polygon(tup)

    return np.sqrt(p.area/np.pi)
