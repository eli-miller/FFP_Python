from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import sys
import numbers
from cmcrameri import cm


def FFP(
    zm=None,
    z0=None,
    umean=None,
    h=None,
    ol=None,
    sigmav=None,
    ustar=None,
    wind_dir=None,
    rs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    rslayer=0,
    nx=1000,
    crop=False,
    fig=False,
    **kwargs
):
    """
    Derives a flux footprint estimate based on the simple parameterization FFP.

    Parameters
    ----------
    zm : float
        Measurement height above displacement height (z-d) [m].
    z0 : float or None
        Roughness length [m]; set to None if not known. Either z0 or `umean` is required.
    umean : float or None
        Mean wind speed at `zm` [m/s]; set to None if not known. If both `z0` and `umean`
        are provided, z0 is used to calculate the footprint.
    h : float
        Boundary layer height [m].
    ol : float
        Obukhov length [m].
    sigmav : float
        Standard deviation of lateral velocity fluctuations [m/s].
    ustar : float
        Friction velocity [m/s].

    Optional Parameters
    -------------------
    wind_dir : float, optional
        Wind direction in degrees (0-360) for rotation of the footprint. Default is None.
    rs : float, list, or None, optional
        Percentage of source area for contour outputs. Values must range between 10% and 90%.
        Can be a single value (e.g., 80) or a list of values (e.g., [10, 20, 30]).
        Expressed as percentages (e.g., 80) or fractions (e.g., 0.8). Default is [10, 20, ..., 80].
    nx : int, optional
        Number of grid elements for the scaled footprint. Higher values increase spatial
        resolution and computation time. Default is 1000. Must be >= 600.
    rslayer : int, optional
        Set to 1 to calculate footprint within the roughness sublayer. This gives a rough
        estimate only, as the model is not valid in this regime. Requires `z0`.
        Default is 0 (no footprint calculated in roughness sublayer).
    crop : int, optional
        Set to 1 to crop output area to the size of the 80% footprint or the largest `rs` provided.
    fig : int, optional
        Set to 1 to display an example plot of the footprint. Default is 0 (no figure).

    Returns
    -------
    x_ci_max : float
        x-location of footprint peak (distance from measurement) [m].
    x_ci : ndplt.pcolormesh(x_2d, y_2d, f_2d)array
        x-array of crosswind-integrated footprint [m].
    f_ci : ndarray
        Footprint function values of the crosswind-integrated footprint [m^-1].
    x_2d : ndarray
        x-grid of 2D footprint [m], rotated if `wind_dir` is provided.
    y_2d : ndarray
        y-grid of 2D footprint [m], rotated if `wind_dir` is provided.
    f_2d : ndarray
        Footprint function values of the 2D footprint [m^-2].
    rs : list or None
        Percentage of footprint contour outputs, if provided in input.
    fr : ndarray or None
        Footprint values for specified `rs`, if provided.
    xr : ndarray or None
        x-array for contour lines of specified `rs`, if provided.
    yr : ndarray or None
        y-array for contour lines of specified `rs`, if provided.
    flag_err : int
        0 if no error occurred; 1 if an error occurred.

    Metadata
    --------
    Created : 15 April 2015, Natascha Kljun.
    Translated to Python : December 2015, Gerardo Fratini (LI-COR Biosciences Inc.).
    Version : 1.42
    Last Updated : 11 December 2019, Gerardo Fratini (Python 3.x port).
    Copyright : 2015 - 2024, Natascha Kljun.
        Reference:
    Kljun, N., Calanca, P., Rotach, M.W., Schmid, H.P., 2015:
    "The simple two-dimensional parameterisation for Flux Footprint Predictions FFP."
    Geosci. Model Dev., 8, 3695-3713. doi:10.5194/gmd-8-3695-2015.

    Contact: natascha.kljun@cec.lu.se
    """

    # ===========================================================================
    # Get kwargs
    show_heatmap = kwargs.get("show_heatmap", True)

    # ===========================================================================
    ## Input check
    flag_err = 0

    ## Check existence of required input pars
    if None in [zm, h, ol, sigmav, ustar] or (z0 is None and umean is None):
        raise_ffp_exception(1)

    # Define rslayer if not passed
    if rslayer == None:
        rslayer == 0

    # Define crop if not passed
    if crop == None:
        crop == 0

    # Define fig if not passed
    if fig == None:
        fig == 0

    # Check passed values
    if zm <= 0.0:
        raise_ffp_exception(2)
    if z0 is not None and umean is None and z0 <= 0.0:
        raise_ffp_exception(3)
    if h <= 10.0:
        raise_ffp_exception(4)
    if zm > h:
        raise_ffp_exception(5)
    if z0 is not None and umean is None and zm <= 12.5 * z0:
        if rslayer is 1:
            raise_ffp_exception(6)
        else:
            raise_ffp_exception(12)
    if float(zm) / ol <= -15.5:
        raise_ffp_exception(7)
    if sigmav <= 0:
        raise_ffp_exception(8)
    if ustar <= 0.1:
        raise_ffp_exception(9)
    if wind_dir is not None:
        if wind_dir > 360 or wind_dir < 0:
            raise_ffp_exception(10)
    if nx < 600:
        raise_ffp_exception(11)

    # Resolve ambiguity if both z0 and umean are passed (defaults to using z0)
    if None not in [z0, umean]:
        raise_ffp_exception(13)

    # ===========================================================================
    # Handle rs
    if rs is not None:
        # Check that rs is a list, otherwise make it a list
        if isinstance(rs, numbers.Number):
            if 0.9 < rs <= 1 or 90 < rs <= 100:
                rs = 0.9
            rs = [rs]
        if not isinstance(rs, list):
            raise_ffp_exception(14)

        # If rs is passed as percentages, normalize to fractions of one
        if np.max(rs) >= 1:
            rs = [x / 100.0 for x in rs]

        # Eliminate any values beyond 0.9 (90%) and inform user
        if np.max(rs) > 0.9:
            raise_ffp_exception(15)
            rs = [item for item in rs if item <= 0.9]

        # Sort levels in ascending order
        rs = list(np.sort(rs))

    # ===========================================================================
    # Model parameters
    a = 1.4524
    b = -1.9914
    c = 1.4622
    d = 0.1359
    ac = 2.17
    bc = 1.66
    cc = 20.0

    xstar_end = 30
    oln = 5000  # limit to L for neutral scaling
    k = 0.4  # von Karman

    # ===========================================================================
    # Scaled X* for crosswind integrated footprint
    xstar_ci_param = np.linspace(d, xstar_end, nx + 2)
    xstar_ci_param = xstar_ci_param[1:]

    # Crosswind integrated scaled F*
    fstar_ci_param = a * (xstar_ci_param - d) ** b * np.exp(-c / (xstar_ci_param - d))
    ind_notnan = ~np.isnan(fstar_ci_param)
    fstar_ci_param = fstar_ci_param[ind_notnan]
    xstar_ci_param = xstar_ci_param[ind_notnan]

    # Scaled sig_y*
    sigystar_param = ac * np.sqrt(bc * xstar_ci_param**2 / (1 + cc * xstar_ci_param))

    # ===========================================================================
    # Real scale x and f_ci
    if z0 is not None:
        # Use z0
        if ol <= 0 or ol >= oln:
            xx = (1 - 19.0 * zm / ol) ** 0.25
            psi_f = (
                np.log((1 + xx**2) / 2.0)
                + 2.0 * np.log((1 + xx) / 2.0)
                - 2.0 * np.arctan(xx)
                + np.pi / 2
            )
        elif ol > 0 and ol < oln:
            psi_f = -5.3 * zm / ol

        x = xstar_ci_param * zm / (1.0 - (zm / h)) * (np.log(zm / z0) - psi_f)
        if np.log(zm / z0) - psi_f > 0:
            x_ci = x
            f_ci = fstar_ci_param / zm * (1.0 - (zm / h)) / (np.log(zm / z0) - psi_f)
        else:
            x_ci_max, x_ci, f_ci, x_2d, y_2d, f_2d = None
            flag_err = 1
    else:
        # Use umean if z0 not available
        x = xstar_ci_param * zm / (1.0 - zm / h) * (umean / ustar * k)
        if umean / ustar > 0:
            x_ci = x
            f_ci = fstar_ci_param / zm * (1.0 - zm / h) / (umean / ustar * k)
        else:
            x_ci_max, x_ci, f_ci, x_2d, y_2d, f_2d = None
            flag_err = 1

    # Maximum location of influence (peak location)
    xstarmax = -c / b + d
    if z0 is not None:
        x_ci_max = xstarmax * zm / (1.0 - (zm / h)) * (np.log(zm / z0) - psi_f)
    else:
        x_ci_max = xstarmax * zm / (1.0 - (zm / h)) * (umean / ustar * k)

    # Real scale sig_y
    if abs(ol) > oln:
        ol = -1e6
    if ol <= 0:  # convective
        scale_const = 1e-5 * abs(zm / ol) ** (-1) + 0.80
    elif ol > 0:  # stable
        scale_const = 1e-5 * abs(zm / ol) ** (-1) + 0.55
    if scale_const > 1:
        scale_const = 1.0
    sigy = sigystar_param / scale_const * zm * sigmav / ustar
    sigy[sigy < 0] = np.nan

    # Real scale f(x,y)
    dx = x_ci[2] - x_ci[1]
    y_pos = np.arange(0, (len(x_ci) / 2.0) * dx * 1.5, dx)
    # f_pos = np.full((len(f_ci), len(y_pos)), np.nan)
    f_pos = np.empty((len(f_ci), len(y_pos)))
    f_pos[:] = np.nan
    for ix in range(len(f_ci)):
        f_pos[ix, :] = (
            f_ci[ix]
            * 1
            / (np.sqrt(2 * np.pi) * sigy[ix])
            * np.exp(-(y_pos**2) / (2 * sigy[ix] ** 2))
        )

    # Complete footprint for negative y (symmetrical)
    y_neg = -np.fliplr(y_pos[None, :])[0]
    f_neg = np.fliplr(f_pos)
    y = np.concatenate((y_neg[0:-1], y_pos))
    f = np.concatenate((f_neg[:, :-1].T, f_pos.T)).T

    # Matrices for output
    x_2d = np.tile(x[:, None], (1, len(y)))
    y_2d = np.tile(y.T, (len(x), 1))
    f_2d = f

    # ===========================================================================
    # Derive footprint ellipsoid incorporating R% of the flux, if requested,
    # starting at peak value.
    dy = dx
    if rs is not None:
        clevs = get_contour_levels(f_2d, dx, dy, rs)
        frs = [item[2] for item in clevs]
        xrs = []
        yrs = []
        for ix, fr in enumerate(frs):
            xr, yr = get_contour_vertices(x_2d, y_2d, f_2d, fr)
            if xr is None:
                frs[ix] = None
            xrs.append(xr)
            yrs.append(yr)
    else:
        if crop:
            rs_dummy = 0.8  # crop to 80%
            clevs = get_contour_levels(f_2d, dx, dy, rs_dummy)
            xrs = []
            yrs = []
            xrs, yrs = get_contour_vertices(x_2d, y_2d, f_2d, clevs[0][2])

    # ===========================================================================
    # Crop domain and footprint to the largest rs value
    if crop:
        xrs_crop = [x for x in xrs if x is not None]
        yrs_crop = [x for x in yrs if x is not None]
        if rs is not None:
            dminx = np.floor(min(xrs_crop[-1]))
            dmaxx = np.ceil(max(xrs_crop[-1]))
            dminy = np.floor(min(yrs_crop[-1]))
            dmaxy = np.ceil(max(yrs_crop[-1]))
        else:
            dminx = np.floor(min(xrs_crop))
            dmaxx = np.ceil(max(xrs_crop))
            dminy = np.floor(min(yrs_crop))
            dmaxy = np.ceil(max(yrs_crop))
        jrange = np.where((y_2d[0] >= dminy) & (y_2d[0] <= dmaxy))[0]
        jrange = np.concatenate(([jrange[0] - 1], jrange, [jrange[-1] + 1]))
        jrange = jrange[np.where((jrange >= 0) & (jrange <= y_2d.shape[0] - 1))[0]]
        irange = np.where((x_2d[:, 0] >= dminx) & (x_2d[:, 0] <= dmaxx))[0]
        irange = np.concatenate(([irange[0] - 1], irange, [irange[-1] + 1]))
        irange = irange[np.where((irange >= 0) & (irange <= x_2d.shape[1] - 1))[0]]
        jrange = [[it] for it in jrange]
        x_2d = x_2d[irange, jrange]
        y_2d = y_2d[irange, jrange]
        f_2d = f_2d[irange, jrange]

    # ===========================================================================
    # Rotate 3d footprint if requested
    if wind_dir is not None:
        wind_dir = wind_dir * np.pi / 180.0
        dist = np.sqrt(x_2d**2 + y_2d**2)
        angle = np.arctan2(y_2d, x_2d)
        x_2d = dist * np.sin(wind_dir - angle)
        y_2d = dist * np.cos(wind_dir - angle)

        if rs is not None:
            for ix, r in enumerate(rs):
                xr_lev = np.array([x for x in xrs[ix] if x is not None])
                yr_lev = np.array([x for x in yrs[ix] if x is not None])
                dist = np.sqrt(xr_lev**2 + yr_lev**2)
                angle = np.arctan2(yr_lev, xr_lev)
                xr = dist * np.sin(wind_dir - angle)
                yr = dist * np.cos(wind_dir - angle)
                xrs[ix] = list(xr)
                yrs[ix] = list(yr)

    # ===========================================================================
    # Plot footprint
    if fig:
        fig_out, ax = plot_footprint(
            x_2d=x_2d, y_2d=y_2d, fs=f_2d, show_heatmap=show_heatmap, clevs=frs
        )

    # ===========================================================================
    # Fill output structure
    if rs is not None:
        return {
            "x_ci_max": x_ci_max,
            "x_ci": x_ci,
            "f_ci": f_ci,
            "x_2d": x_2d,
            "y_2d": y_2d,
            "f_2d": f_2d,
            "rs": rs,
            "fr": frs,
            "xr": xrs,
            "yr": yrs,
            "flag_err": flag_err,
        }
    else:
        return {
            "x_ci_max": x_ci_max,
            "x_ci": x_ci,
            "f_ci": f_ci,
            "x_2d": x_2d,
            "y_2d": y_2d,
            "f_2d": f_2d,
            "flag_err": flag_err,
        }


# ===============================================================================
# ===============================================================================
def get_contour_levels(
    footprint_values, grid_spacing_x, grid_spacing_y, contour_percentages=None
):
    """
    Calculate contour levels for given footprint values and contour percentages.

    Parameters
    ----------
    footprint_values : ndarray
        2D array of footprint values.
    grid_spacing_x : float
        Grid spacing in the x-direction.
    grid_spacing_y : float
        Grid spacing in the y-direction.
    contour_percentages : int, float, or list, optional
        Contour percentages to calculate levels for. If not provided, defaults to [10%, 20%, ..., 90%].

    Returns
    -------
    list of tuples
        Each tuple contains (percentage, area, contour_level) for the given contour percentages.
    """

    import numpy as np
    from numpy import ma
    import sys

    # Check input and resolve to default levels if needed
    if not isinstance(contour_percentages, (int, float, list)):
        contour_percentages = list(np.linspace(0.10, 0.90, 9))
    if isinstance(contour_percentages, (int, float)):
        contour_percentages = [contour_percentages]

    # Levels
    contour_levels = np.empty(len(contour_percentages))
    contour_levels[:] = np.nan
    areas = np.empty(len(contour_percentages))
    areas[:] = np.nan

    sorted_values = np.sort(footprint_values, axis=None)[::-1]
    masked_sorted_values = ma.masked_array(
        sorted_values, mask=(np.isnan(sorted_values) | np.isinf(sorted_values))
    )  # Masked array for handling potential nan

    cumulative_sum = (
        masked_sorted_values.cumsum().filled(np.nan) * grid_spacing_x * grid_spacing_y
    )
    for index, percentage in enumerate(contour_percentages):
        difference_cumulative_sum = np.abs(cumulative_sum - percentage)
        contour_levels[index] = sorted_values[np.nanargmin(difference_cumulative_sum)]
        areas[index] = cumulative_sum[np.nanargmin(difference_cumulative_sum)]

    return [
        (round(percentage, 3), area, contour_level)
        for percentage, area, contour_level in zip(
            contour_percentages, areas, contour_levels
        )
    ]


# ===============================================================================
def get_contour_vertices(x, y, f, lev):
    cs = plt.contour(x, y, f, [lev])
    plt.close()

    if not cs.collections:
        return [None, None]  # Return if no contour lines were found

    # Process all segments from the first contour line
    path = cs.collections[0].get_paths()
    if not path:
        return [None, None]  # Return if there are no paths in the collection

    segs = path[0].vertices  # Take the first path's vertices
    if segs.size == 0:
        return [None, None]

    xr, yr = segs[:, 0], segs[:, 1]

    # Check if the contour reaches the boundary of the physical domain
    if (
        np.min(xr) <= x.min()
        or np.max(xr) >= x.max()
        or np.min(yr) <= y.min()
        or np.max(yr) >= y.max()
    ):
        return [None, None]

    return [xr, yr]  # x, y coordinates of contour points.


# ===============================================================================
def plot_footprint(
    x_2d,
    y_2d,
    fs,
    clevs=None,
    show_heatmap=True,
    normalize=None,
    colormap=cm.batlow,
    line_width=0.5,
    iso_labels=None,
):
    """Plot footprint function and contours if request"""

    import numpy as np
    import matplotlib.pyplot as plt

    # import matplotlib.cm as cm

    from matplotlib.colors import LogNorm

    # If input is a list of footprints, don't show footprint but only contours,
    # with different colors
    if isinstance(fs, list):
        show_heatmap = False
    else:
        fs = [fs]

    # Define colors for each contour set
    cs = [colormap(ix) for ix in np.linspace(0, 1, len(fs))]

    # Initialize figure
    fig, ax = plt.subplots(figsize=(12, 10))
    # fig.patch.set_facecolor('none')
    # ax.patch.set_facecolor('none')

    if clevs is not None:
        # Temporary patch for pyplot.contour requiring contours to be in ascending orders
        clevs = clevs[::-1]

        # Eliminate contour levels that were set to None
        # (e.g. because they extend beyond the defined domain)
        clevs = [clev for clev in clevs if clev is not None]

        # Plot contour levels of all passed footprints
        # Plot isopleth
        levs = [clev for clev in clevs]
        for f, c in zip(fs, cs):
            cc = [c] * len(levs)
            if show_heatmap:
                cp = ax.contour(x_2d, y_2d, f, levs, colors="w", linewidths=line_width)
            else:
                cp = ax.contour(x_2d, y_2d, f, levs, colors=cc, linewidths=line_width)
            # Isopleth Labels
            if iso_labels is not None:
                pers = [str(int(clev[0] * 100)) + "%" for clev in clevs]
                fmt = {}
                for l, s in zip(cp.levels, pers):
                    fmt[l] = s
                plt.clabel(cp, cp.levels[:], inline=1, fmt=fmt, fontsize=7)

    # plot footprint heatmap if requested and if only one footprint is passed
    if show_heatmap:
        if normalize == "log":
            norm = LogNorm()
        else:
            norm = None

        for f in fs:
            pcol = plt.pcolormesh(x_2d, y_2d, f, cmap=colormap, norm=norm)
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.gca().set_aspect("equal", "box")

        cbar = fig.colorbar(pcol, shrink=1.0, format="%.3e")
        # cbar.set_label('Flux contribution', color = 'k')
    plt.show()

    return fig, ax


# ===============================================================================
# ===============================================================================
exTypes = {
    "message": "Message",
    "alert": "Alert",
    "error": "Error",
    "fatal": "Fatal error",
}

exceptions = [
    {
        "code": 1,
        "type": exTypes["fatal"],
        "msg": "At least one required parameter is missing. Please enter all "
        "required inputs. Check documentation for details.",
    },
    {
        "code": 2,
        "type": exTypes["error"],
        "msg": "zm (measurement height) must be larger than zero.",
    },
    {
        "code": 3,
        "type": exTypes["error"],
        "msg": "z0 (roughness length) must be larger than zero.",
    },
    {
        "code": 4,
        "type": exTypes["error"],
        "msg": "h (BPL height) must be larger than 10 m.",
    },
    {
        "code": 5,
        "type": exTypes["error"],
        "msg": "zm (measurement height) must be smaller than h (PBL height).",
    },
    {
        "code": 6,
        "type": exTypes["alert"],
        "msg": "zm (measurement height) should be above roughness sub-layer (12.5*z0).",
    },
    {
        "code": 7,
        "type": exTypes["error"],
        "msg": "zm/ol (measurement height to Obukhov length ratio) must be equal or larger than -15.5.",
    },
    {
        "code": 8,
        "type": exTypes["error"],
        "msg": "sigmav (standard deviation of crosswind) must be larger than zero.",
    },
    {
        "code": 9,
        "type": exTypes["error"],
        "msg": "ustar (friction velocity) must be >=0.1.",
    },
    {
        "code": 10,
        "type": exTypes["error"],
        "msg": "wind_dir (wind direction) must be >=0 and <=360.",
    },
    {
        "code": 11,
        "type": exTypes["fatal"],
        "msg": "Passed data arrays (ustar, zm, h, ol) don't all have the same length.",
    },
    {
        "code": 12,
        "type": exTypes["fatal"],
        "msg": "No valid zm (measurement height above displacement height) passed.",
    },
    {
        "code": 13,
        "type": exTypes["alert"],
        "msg": "Using z0, ignoring umean if passed.",
    },
    {"code": 14, "type": exTypes["alert"], "msg": "No valid z0 passed, using umean."},
    {"code": 15, "type": exTypes["fatal"], "msg": "No valid z0 or umean array passed."},
    {
        "code": 16,
        "type": exTypes["error"],
        "msg": "At least one required input is invalid. Skipping current footprint.",
    },
    {
        "code": 17,
        "type": exTypes["alert"],
        "msg": "Only one value of zm passed. Using it for all footprints.",
    },
    {
        "code": 18,
        "type": exTypes["fatal"],
        "msg": "if provided, rs must be in the form of a number or a list of numbers.",
    },
    {
        "code": 19,
        "type": exTypes["alert"],
        "msg": "rs value(s) larger than 90% were found and eliminated.",
    },
    {
        "code": 20,
        "type": exTypes["error"],
        "msg": "zm (measurement height) must be above roughness sub-layer (12.5*z0).",
    },
]


def raise_ffp_exception(code):
    """Raise exception or prints message according to specified code"""

    ex = [it for it in exceptions if it["code"] == code][0]
    string = ex["type"] + "(" + str(ex["code"]).zfill(4) + "):\n " + ex["msg"]

    print("")
    if ex["type"] == exTypes["fatal"]:
        string = string + "\n FFP_fixed_domain execution aborted."
        raise Exception(string)
    else:
        print(string)
