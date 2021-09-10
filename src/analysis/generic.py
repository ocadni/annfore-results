import numpy as np

def filter_nans(array):
    idx = np.isnan(array)
    return array[np.logical_not(idx)]

def histo_points(data, bins=50, **kwargs):
    y,x = np.histogram(data, bins=bins, **kwargs)
    steps = (x-np.roll(x,1))[1:]
    xpts = x[:-1]+steps/2

    return xpts, y

def make_histo_scaled_cum(pos, xscale=1, nbins=50, range_=(0,1)):
    """
    Make cumulative histogram scaling x axis and y axis
    """
    pos_s = pos.flatten()
    num_inst = len(pos_s)
    res =np.histogram(pos_s/xscale, bins=nbins, range=range_)
    y_plt = np.insert(np.cumsum(res[0])/num_inst,0,0)
    return res[1], y_plt