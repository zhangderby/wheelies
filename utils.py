from prysm.mathops import np, _np
from skimage.transform import downscale_local_mean

def radial_profile(data, center=None):

    # Define the center of the array if not specified
    if center is None:
        center = (data.shape[0] // 2, data.shape[1] // 2)
        
    # Get the 2D grid of coordinates
    y, x = np.indices(data.shape)
    
    # Calculate the Euclidean distance from each pixel to the center
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    # Convert distances to integer bin indices
    r_int = r.astype(int)
    
    # Sum up the data values in each radial bin
    radial_sum = np.bincount(r_int.ravel(), weights=data.ravel())
    
    # Count the number of pixels falling into each radial bin
    radial_counts = np.bincount(r_int.ravel())
    
    # Calculate the average (avoiding division by zero)
    profile = np.divide(radial_sum, radial_counts, out=np.zeros_like(radial_sum))
    
    return profile

def make_pinwheel(Dpup, Npup, rings, segments_ring1, segment_separation):

    NsegR1 = segments_ring1
    Nsegs = 1 + np.array([segments_ring1 * (i + 1) for i in range(rings)]).sum()
    Dsep = segment_separation / Dpup

    Q = 4

    x = np.linspace(-0.5, 0.5, Npup * Q)
    x, y = np.meshgrid(x, x)
    r = np.sqrt(x ** 2 + y ** 2)
    t = np.arctan2(y, x)

    pw = np.zeros_like(x)

    r_rings = np.array([(i * 2 + 1) / ((rings * 2 + 1) * 2) for i in range(rings + 1)])

    chord = r_rings[1] - r_rings[0]
    theta = np.pi / (Nsegs - 1)
    r_seg = chord / 2 / np.sin(theta / 2)

    m = r < (r_rings[0] - Dsep / 2)
    pw[m] = 1

    for ring in range(rings):
        for seg in range((ring + 1) * NsegR1):
    
            # inner and outer bounds
            m = (r > (r_rings[ring] + Dsep / 2)) & (r < (r_rings[ring + 1] - Dsep / 2))

            # side bounds
            NsegR = NsegR1 * (ring + 1)
            adj = (r_rings[ring + 1] + r_rings[ring]) / 2
            opp = np.sqrt(r_seg ** 2 - (chord ** 2 / 4))
            hyp = np.sqrt(adj ** 2 + opp ** 2)

            tSeg = seg * (2 * np.pi / NsegR)
            t1 = tSeg + (np.pi / NsegR)
            t2 = np.arcsin(opp / hyp)
            t3 = t1 + t2
            xc = np.cos(t3) * hyp
            yc = np.sin(t3) * hyp
            rc = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
            m = m & (rc < r_seg - Dsep / 2)

            tSeg = seg * (2 * np.pi / NsegR)
            t1 = tSeg - (np.pi / NsegR)
            t2 = np.arcsin(opp / hyp)
            t3 = t1 + t2
            xc = np.cos(t3) * hyp
            yc = np.sin(t3) * hyp
            rc = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
            m = m & (rc > r_seg + Dsep / 2)

            pw[m] = 1

    if isinstance(pw, _np.ndarray):
        pw = downscale_local_mean(pw, (Q, Q))
    else:
        pw = np.array(downscale_local_mean(pw.get(), (Q, Q)))        

    return pw

def make_pinwheel_eac2(Npup, segment_separation):

    Dpup = 6e3
    rings = 1
    segments_ring1 = 6

    NsegR1 = segments_ring1
    Nsegs = 1 + np.array([segments_ring1 * (i + 1) for i in range(rings)]).sum()
    Dsep = segment_separation / Dpup

    Q = 4

    x = np.linspace(-0.5, 0.5, Npup * Q)
    x, y = np.meshgrid(x, x)
    r = np.sqrt(x ** 2 + y ** 2)
    t = np.arctan2(y, x)

    pw = np.zeros_like(x)

    r_rings = np.zeros(2)
    r_rings[0] = 0.25
    r_rings[1] = 0.5

    chord = r_rings[1] - r_rings[0]
    theta = np.pi / (Nsegs - 1)
    r_seg = chord / 2 / np.sin(theta / 2)

    m = r < (r_rings[0] - Dsep / 2)
    pw[m] = 1

    for ring in range(rings):
        for seg in range((ring + 1) * NsegR1):
    
            # inner and outer bounds
            m = (r > (r_rings[ring] + Dsep / 2)) & (r < (r_rings[ring + 1] - Dsep / 2))

            # side bounds
            NsegR = NsegR1 * (ring + 1)
            adj = (r_rings[ring + 1] + r_rings[ring]) / 2
            opp = np.sqrt(r_seg ** 2 - (chord ** 2 / 4))
            hyp = np.sqrt(adj ** 2 + opp ** 2)

            tSeg = seg * (2 * np.pi / NsegR)
            t1 = tSeg + (np.pi / NsegR)
            t2 = np.arcsin(opp / hyp)
            t3 = t1 + t2
            xc = np.cos(t3) * hyp
            yc = np.sin(t3) * hyp
            rc = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
            m = m & (rc < r_seg - Dsep / 2)

            tSeg = seg * (2 * np.pi / NsegR)
            t1 = tSeg - (np.pi / NsegR)
            t2 = np.arcsin(opp / hyp)
            t3 = t1 + t2
            xc = np.cos(t3) * hyp
            yc = np.sin(t3) * hyp
            rc = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
            m = m & (rc > r_seg + Dsep / 2)

            pw[m] = 1

    if isinstance(pw, _np.ndarray):
        pw = downscale_local_mean(pw, (Q, Q))
    else:
        pw = np.array(downscale_local_mean(pw.get(), (Q, Q)))        

    return pw