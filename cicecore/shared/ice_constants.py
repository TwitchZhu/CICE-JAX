"""This module defines a variety of physical and numerical constants
used throughout the ice model.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

author: Elizabeth C. Hunke, LANL
"""

from cicecore.shared.ice_kinds_mod import dbl_kind, int_kind, real_kind

#-----------------------------------------------------------------
# physical constants
#-----------------------------------------------------------------

omega = dbl_kind(7.292e-5)   # angular velocity of earth (rad/sec)
radius = dbl_kind(6.37e6)      # earth radius (m)
spval_dbl = dbl_kind(1.0e30)   # special value (double precision)
spval = real_kind(1.0e30)      # special value for netCDF output

# these are currently set so as to have no effect on the decomposition
shlat = dbl_kind(30.0)         # artificial masking edge (deg)
nhlat = dbl_kind(-30.0)        # artificial masking edge (deg)

#-----------------------------------------------------------------
# numbers used outside the column package
#-----------------------------------------------------------------

c0 = dbl_kind(0.0)
c1 = dbl_kind(1.0)
c1p5 = dbl_kind(1.5)
c2 = dbl_kind(2.0)
c3 = dbl_kind(3.0)
c4 = dbl_kind(4.0)
c5 = dbl_kind(5.0)
c6 = dbl_kind(6.0)
c8 = dbl_kind(8.0)
c9 = dbl_kind(9.0)
c10 = dbl_kind(10.0)
c12 = dbl_kind(12.0)
c15 = dbl_kind(15.0)
c16 = dbl_kind(16.0)
c20 = dbl_kind(20.0)
c25 = dbl_kind(25.0)
c30 = dbl_kind(30.0)
c100 = dbl_kind(100.0)
c180 = dbl_kind(180.0)
c360 = dbl_kind(360.0)
c365 = dbl_kind(365.0)
c400 = dbl_kind(400.0)
c1000 = dbl_kind(1000.0)
c3600 = dbl_kind(3600.0)
p001 = dbl_kind(0.001)
p01 = dbl_kind(0.01)
p025 = dbl_kind(0.025)
p05 = dbl_kind(0.05)
p1 = dbl_kind(0.1)
p15 = dbl_kind(0.15)
p2 = dbl_kind(0.2)
p25 = dbl_kind(0.25)
p3 = dbl_kind(0.3)
p4 = dbl_kind(0.4)
p5 = dbl_kind(0.5)
p6 = dbl_kind(0.6)
p75 = dbl_kind(0.75)
p111 = c1 / c9
p166 = c1 / c6
p222 = c2 / c9
p333 = c1 / c3
p666 = c2 / c3
p055 = p111 * p5
p027 = p055 * p5
eps04 = dbl_kind(1.0e-4)
eps13 = dbl_kind(1.0e-13)
eps16 = dbl_kind(1.0e-16)

#-----------------------------------------------------------------
# location of fields for staggered grids
#-----------------------------------------------------------------

field_loc_unknown = int_kind(0)
field_loc_noupdate = int_kind(-1)
field_loc_center = int_kind(1)
field_loc_NEcorner = int_kind(2)
field_loc_Nface = int_kind(3)
field_loc_Eface = int_kind(4)

#-----------------------------------------------------------------
# field type attribute - necessary for handling
# changes of direction across tripole boundary
#-----------------------------------------------------------------

field_type_unknown = int_kind(0)
field_type_noupdate = int_kind(-1)
field_type_scalar = int_kind(1)
field_type_vector = int_kind(2)
field_type_angle = int_kind(3)

#-----------------------------------------------------------------
# conversion factors
#-----------------------------------------------------------------

cm_to_m = dbl_kind(0.01)       # cm to meters
m_to_cm = dbl_kind(100.0)      # meters to cm
m2_to_km2 = dbl_kind(1.0e-6)   # m^2 to km^2
kg_to_g = dbl_kind(1000.0)     # kilograms to grams
mps_to_cmpdy = dbl_kind(8.64e6)  # m per s to cm per day


def ice_init_constants(
    omega_in=None,
    radius_in=None,
    spval_dbl_in=None,
    spval_in=None,
    shlat_in=None,
    nhlat_in=None,
):
    global omega, radius, spval_dbl, spval, shlat, nhlat
    if omega_in is not None:
        omega = dbl_kind(omega_in)
    if radius_in is not None:
        radius = dbl_kind(radius_in)
    if spval_dbl_in is not None:
        spval_dbl = dbl_kind(spval_dbl_in)
    if spval_in is not None:
        spval = real_kind(spval_in)
    if shlat_in is not None:
        shlat = dbl_kind(shlat_in)
    if nhlat_in is not None:
        nhlat = dbl_kind(nhlat_in)


def ice_query_constants():
    return {
        "omega": omega,
        "radius": radius,
        "spval_dbl": spval_dbl,
        "spval": spval,
        "shlat": shlat,
        "nhlat": nhlat,
    }
