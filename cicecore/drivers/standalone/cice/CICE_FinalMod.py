"""Finalize the standalone CICE run, from CICE_FinalMod.F90."""

from cicecore.cicedyn.general import ice_init
from cicecore.cicedyn.infrastructure.comm.serial import ice_timers
from cicecore.shared import ice_fileunits


def CICE_Finalize():
    """Stop the run timer and write the standalone completion message."""
    #-------------------------------------------------------------------
    # stop timers and print timer info
    #-------------------------------------------------------------------
    ice_timers.ice_timer_stop(ice_timers.timer_total)
    ice_timers.ice_timer_print_all(stats=bool(ice_init.timer_stats))

    # Icepack failures are raised after each JAX time step. The single
    # Python process has no MPI exit or Fortran file-unit table to release.
    nu_diag = ice_fileunits.nu_diag
    nu_diag.write(" \n")
    nu_diag.write(" CICE COMPLETED SUCCESSFULLY \n")
    nu_diag.write(" \n")
    nu_diag.flush()
