"""This module contains the CICE initialization routine that sets model
parameters and initializes the grid and CICE state variables.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

2008: E. Hunke moved ESMF code to its own driver
2006: Converted to free form source (F90) by Elizabeth Hunke
authors Elizabeth C. Hunke, LANL
        William H. Lipscomb, LANL
        Philip W. Jones, LANL

Column-physics failures raise. The icepack_warnings_flush and abort_ice
calls that follow icepack routines in the Fortran are not ported.
"""

from cicecore.cicedyn.analysis import ice_diagnostics
from cicecore.cicedyn.analysis import ice_history
from cicecore.cicedyn.infrastructure.io.io_netcdf import ice_history_write
from cicecore.cicedyn.infrastructure.io.io_netcdf import ice_restart
from cicecore.cicedyn.dynamics import ice_dyn_eap
from cicecore.cicedyn.dynamics import ice_dyn_evp
from cicecore.cicedyn.dynamics import ice_dyn_shared
from cicecore.cicedyn.dynamics import ice_dyn_vp
from cicecore.cicedyn.dynamics import ice_transport_driver
from cicecore.cicedyn.general import ice_flux
from cicecore.cicedyn.general import ice_forcing
from cicecore.cicedyn.general import ice_forcing_bgc
from cicecore.cicedyn.general import ice_flux_bgc
from cicecore.cicedyn.general import ice_init
from cicecore.cicedyn.general import ice_state
from cicecore.cicedyn.infrastructure import ice_domain
from cicecore.cicedyn.infrastructure import ice_grid
from cicecore.cicedyn.infrastructure import ice_restart_driver
from cicecore.cicedyn.infrastructure import ice_restoring
from cicecore.cicedyn.infrastructure.comm.serial import ice_timers
from cicecore.shared import ice_arrays_column
from cicecore.shared import ice_calendar
from cicecore.shared import ice_domain_size
from cicecore.shared import ice_fileunits
from cicecore.shared import ice_init_column
from cicecore.shared import ice_restart_column
from cicecore.shared.ice_constants import c0
from icepack.columnphysics import icepack_fsd
from icepack.columnphysics import icepack_intfc
from icepack.columnphysics import icepack_itd
from icepack.columnphysics import icepack_parameters
from icepack.columnphysics import icepack_shortwave
from icepack.columnphysics import icepack_snow
from icepack.columnphysics import icepack_tracers


def CICE_Initialize():
    """Initialize the basic state, grid and all necessary parameters for
    running the CICE model.  Return the initial state in routine
    export state.
    Note: This initialization driver is designed for standalone and
          CESM-coupled applications.  For other
          applications (e.g., standalone CAM), this driver would be
          replaced by a different driver that calls subroutine cice_init,
          where most of the work is done.
    """
    #--------------------------------------------------------------------
    # model initialization
    #--------------------------------------------------------------------

    cice_init()


def cice_init():
    """Initialize CICE model."""
    # call init_communicate     # initial setup for message passing
    # call init_fileunits       # unit numbers
    # MPI and file-unit numbers are not ported.

    # tcx debug, this will create a different logfile for each pe
    # if (my_task /= master_task) nu_diag = 100+my_task

    icepack_intfc.icepack_configure()  # initialize icepack
    # call icepack_warnings_flush(nu_diag)
    # if (icepack_warnings_aborted()) call abort_ice(trim(subname), &
    #     file=__FILE__,line= __LINE__)

    ice_init.input_data()           # namelist variables
    ice_init_column.input_zbgc()    # vertical biogeochemistry namelist
    ice_init.count_tracers()        # count tracers

    # Call this as early as possible, must be after memory_stats is read
    # if (my_task == master_task) then
    #    call ice_memusage_init(nu_diag)
    #    call ice_memusage_print(nu_diag,subname//':start')
    # endif
    # Memory statistics are not ported.

    ice_domain.init_domain_blocks()  # domain size and boundary types
    ice_grid.init_grid1()           # domain distribution
    ice_grid.alloc_grid()           # allocate grid arrays
    ice_arrays_column.alloc_arrays_column()  # allocate column arrays
    ice_state.alloc_state()         # allocate state arrays
    ice_flux_bgc.alloc_flux_bgc()   # allocate flux_bgc arrays
    ice_flux.alloc_flux()           # allocate flux arrays
    ice_timers.init_ice_timers()  # initialize all timers
    ice_timers.ice_timer_start(ice_timers.timer_total)  # start timing entire run
    ice_grid.init_grid2()           # grid variables
    ice_init_column.init_zbgc()         # vertical biogeochemistry initialization
    ice_calendar.init_calendar()        # initialize some calendar stuff
    ice_history.init_hist(ice_calendar.dt)  # initialize output history file

    # These switches are also needed when no momentum solver is initialized.
    kdyn = int(ice_init.kdyn)
    ice_dyn_shared.kdyn = kdyn
    ice_dyn_shared.kridge = int(ice_init.kridge)
    ice_dyn_shared.grid_ice = str(ice_init.grid_ice).strip()
    if kdyn == 1:
        ice_dyn_evp.init_evp()
    elif kdyn == 2:
        ice_dyn_eap.init_eap()          # define eap dynamics parameters, variables
    elif kdyn == 3:
        ice_dyn_vp.init_vp()           # define vp dynamics parameters, variables

    ice_flux.init_coupler_flux()    # initialize fluxes exchanged with coupler

    ice_init_column.init_thermo_vertical() # initialize vertical thermodynamics

    # if (my_task == master_task): one process, so the hist call always runs.
    icepack_itd.icepack_init_itd(hin_max=ice_arrays_column.hin_max)  # ice thickness distribution
    icepack_itd.icepack_init_itd_hist(
        hin_max=ice_arrays_column.hin_max,
        c_hi_range=ice_arrays_column.c_hi_range,
    )  # output

    # call icepack_warnings_flush(nu_diag)
    # if (icepack_warnings_aborted()) call abort_ice(trim(subname), &
    #     file=__FILE__,line= __LINE__)
    tr_fsd = bool(
        icepack_tracers.icepack_query_tracer_flags(tr_fsd_out=None)["tr_fsd"]
    )

    if tr_fsd:
        # call icepack_init_fsd_bounds ( &
        #    floe_rad_l_out = floe_rad_l,    &  # fsd size lower bound in m (radius)
        #    floe_rad_c_out = floe_rad_c,    &  # fsd size bin centre in m (radius)
        #    floe_binwidth_out = floe_binwidth, &  # fsd size bin width in m (radius)
        #    c_fsd_range_out = c_fsd_range,   &  # string for history output
        #    write_diags=(my_task == master_task))  # write diag on master only
        icepack_fsd.icepack_init_fsd_bounds(
            floe_rad_l=ice_arrays_column.floe_rad_l,
            floe_rad_c=ice_arrays_column.floe_rad_c,
            floe_binwidth=ice_arrays_column.floe_binwidth,
            c_fsd_range=ice_arrays_column.c_fsd_range,
            write_diags=True,
        )

    ice_forcing.init_forcing_ocn(ice_calendar.dt) # initialize sss and sst from data
    ice_init.init_state()           # initialize the ice state
    ice_transport_driver.init_transport()       # initialize horizontal transport
    # call ice_HaloRestore_init ! restored boundary conditions
    # Returns immediately when restore_ice is false.
    ice_restoring.ice_HaloRestore_init()

    queried = icepack_parameters.icepack_query_parameters(
        skl_bgc_out=None,
        z_tracers_out=None,
        wave_spec_out=None,
        snw_aging_table_out=None,
    )
    # call icepack_warnings_flush(nu_diag)
    # if (icepack_warnings_aborted()) call abort_ice(trim(subname), &
    #     file=__FILE__,line= __LINE__)
    skl_bgc = bool(queried["skl_bgc"])
    z_tracers = bool(queried["z_tracers"])
    wave_spec = bool(queried["wave_spec"])

    if skl_bgc or z_tracers:
        # call alloc_forcing_bgc # allocate biogeochemistry arrays
        ice_forcing_bgc.alloc_forcing_bgc()

    init_restart()         # initialize restart variables
    ice_diagnostics.init_diags()           # initialize diagnostic output points
    ice_flux.init_history_therm()   # initialize thermo history variables
    ice_flux.init_history_dyn()     # initialize dynamic history variables
    ice_calendar.calc_timesteps()       # update timestep counter if not using npt_unit="1"
    icepack_shortwave.icepack_init_radiation() # initialize icepack shortwave tables

    # call icepack_warnings_flush(nu_diag)
    # if (icepack_warnings_aborted()) call abort_ice(trim(subname), &
    #     file=__FILE__,line= __LINE__)
    flags = icepack_tracers.icepack_query_tracer_flags(
        tr_aero_out=None,
        tr_zaero_out=None,
        tr_iso_out=None,
        tr_snow_out=None,
    )
    tr_aero = bool(flags["tr_aero"])
    tr_zaero = bool(flags["tr_zaero"])
    tr_iso = bool(flags["tr_iso"])
    tr_snow = bool(flags["tr_snow"])

    # Initialize shortwave components using swdn from previous timestep
    # if restarting. These components will be scaled to current forcing
    # in prep_radiation.
    runtype = str(ice_init.runtype).strip()
    restart = bool(ice_init.restart)
    if runtype == "continue" or restart:
        ice_init_column.init_shortwave()    # initialize radiative transfer

    if ice_calendar.write_ic:
        ice_history.accum_hist(ice_calendar.dt)  # write initial conditions
    ice_history_write.ice_read_hist()  # read history restarts

    # tcraig, use advance_timestep here
    #      istep  = istep  + 1    # update time step counters
    #      istep1 = istep1 + 1
    #      time = time + dt       # determine the time and date
    #      call calendar(time)    # at the end of the first timestep
    ice_calendar.advance_timestep()

    #--------------------------------------------------------------------
    # coupler communication or forcing data initialization
    #--------------------------------------------------------------------

    ice_forcing.init_forcing_atmo()    # initialize atmospheric forcing (standalone)

    if tr_fsd and wave_spec:
        # call get_wave_spec # wave spectrum in ice
        # for now, wave_spectrum is constant in time
        ice_forcing.get_wave_spec()
    ice_forcing.get_forcing_atmo()     # atmospheric forcing from data
    ice_forcing.get_forcing_ocn(ice_calendar.dt)  # ocean forcing from data

    # snow aging lookup table initialization
    if tr_snow:         # advanced snow physics
        # call icepack_init_snow()
        icepack_snow.icepack_init_snow()
        # if (snw_aging_table(1:4) /= 'test') then
        #    call init_snowtable()
        # endif
        snw_aging_table = str(queried["snw_aging_table"])
        if snw_aging_table[:4] != "test":
            ice_forcing.init_snowtable()

    # isotopes
    if tr_iso:
        # call fiso_default                 # default values
        ice_forcing_bgc.fiso_default()

    if tr_aero or tr_zaero:
        # call faero_default    # default values
        # if (tr_aero)  call faero_data                   ! data file
        # if (tr_zaero) call fzaero_data                  ! data file (gx1)
        # Those two data-file calls are commented out in the Fortran.
        ice_forcing_bgc.faero_default()
    if skl_bgc or z_tracers:
        # call get_forcing_bgc  # biogeochemistry
        ice_forcing_bgc.get_forcing_bgc()
    if z_tracers:
        # call get_atm_bgc                   # biogeochemistry
        ice_forcing_bgc.get_atm_bgc()

    if runtype == "initial" and not restart:
        ice_init_column.init_shortwave()    # initialize radiative transfer using current swdn

    ice_flux.init_flux_atm()        # initialize atmosphere fluxes sent to coupler
    ice_flux.init_flux_ocn()        # initialize ocean fluxes sent to coupler

    # call dealloc_grid         # deallocate temporary grid arrays
    # Temporary grid arrays are not deallocated.

    # if (my_task == master_task) then
    #    call ice_memusage_print(nu_diag,subname//':end')
    # endif


def init_restart():
    """Read the restart file when one is requested, then fill tracers that are not read.

    The aggregate over categories runs on ocean cells. Land tracers are set to zero.
    """
    subname = "(init_restart)"
    _ = subname
    ice_restart_column._sync_flags()
    flags = icepack_tracers.icepack_query_tracer_flags(
        tr_iage_out=None,
        tr_FY_out=None,
        tr_lvl_out=None,
        tr_pond_lvl_out=None,
        tr_pond_sealvl_out=None,
        tr_pond_topo_out=None,
        tr_aero_out=None,
        tr_brine_out=None,
        tr_snow_out=None,
        tr_fsd_out=None,
        tr_iso_out=None,
    )
    tr_iage = bool(flags["tr_iage"])
    tr_FY = bool(flags["tr_FY"])
    tr_lvl = bool(flags["tr_lvl"])
    tr_pond_lvl = bool(flags["tr_pond_lvl"])
    tr_pond_sealvl = bool(flags["tr_pond_sealvl"])
    tr_pond_topo = bool(flags["tr_pond_topo"])
    tr_snow = bool(flags["tr_snow"])
    tr_fsd = bool(flags["tr_fsd"])
    tr_iso = bool(flags["tr_iso"])
    tr_aero = bool(flags["tr_aero"])
    tr_brine = bool(flags["tr_brine"])
    queried = icepack_parameters.icepack_query_parameters(
        skl_bgc_out=None,
        z_tracers_out=None,
    )
    skl_bgc = bool(queried["skl_bgc"])
    z_tracers = bool(queried["z_tracers"])
    runtype = str(ice_init.runtype).strip()
    restart = bool(ice_init.restart)
    kdyn = int(ice_init.kdyn)

    if runtype == "continue":
        # start from core restart file
        ice_restart_driver.restartfile()  # given by pointer in ice_in
        ice_calendar.calendar()  # update time parameters
        if kdyn == 2:
            ice_dyn_eap.read_restart_eap()  # EAP
    elif restart:  # ice_ic = core restart file
        ice_restart_driver.restartfile(ice_init.ice_ic)  # or 'default' or 'none'
        # call restartfile_v4 (ice_ic)  ! CICE v4.1 binary restart file
        # if (kdyn == 2) call read_restart_eap

    # tracers
    # ice age tracer
    if tr_iage:
        if runtype == "continue":
            ice_restart_column.restart_age = True
        if ice_restart_column.restart_age:
            ice_restart_column.read_restart_age()
        else:
            ice_init_column.init_age(ice_state_slice("nt_iage"))
    if tr_FY:
        if runtype == "continue":
            ice_restart_column.restart_FY = True
        if ice_restart_column.restart_FY:
            ice_restart_column.read_restart_FY()
        else:
            ice_init_column.init_FY(ice_state_slice("nt_FY"))
    if tr_lvl:
        if runtype == "continue":
            ice_restart_column.restart_lvl = True
        if ice_restart_column.restart_lvl:
            ice_restart_column.read_restart_lvl()
        else:
            ice_init_column.init_lvl(
                ice_state_slice("nt_alvl"),
                ice_state_slice("nt_vlvl"),
            )
    if tr_pond_lvl:
        if runtype == "continue":
            ice_restart_column.restart_pond_lvl = True
        if ice_restart_column.restart_pond_lvl:
            ice_restart_column.read_restart_pond_lvl()
        else:
            ice_init_column.init_meltponds_lvl(
                ice_state_slice("nt_apnd"),
                ice_state_slice("nt_hpnd"),
                ice_state_slice("nt_ipnd"),
                ice_arrays_column.dhsn,
            )
    if tr_pond_sealvl:
        if runtype == "continue":
            ice_restart_column.restart_pond_sealvl = True
        if ice_restart_column.restart_pond_sealvl:
            ice_restart_column.read_restart_pond_sealvl()
        else:
            ice_init_column.init_meltponds_sealvl(
                ice_state_slice("nt_apnd"),
                ice_state_slice("nt_hpnd"),
                ice_state_slice("nt_ipnd"),
                ice_arrays_column.dhsn,
            )
    if tr_pond_topo:
        if runtype == "continue":
            ice_restart_column.restart_pond_topo = True
        if ice_restart_column.restart_pond_topo:
            ice_restart_column.read_restart_pond_topo()
        else:
            ice_init_column.init_meltponds_topo(
                ice_state_slice("nt_apnd"),
                ice_state_slice("nt_hpnd"),
                ice_state_slice("nt_ipnd"),
            )
    if tr_snow:
        if runtype == "continue":
            ice_restart_column.restart_snow = True
        if ice_restart_column.restart_snow:
            ice_restart_column.read_restart_snow()
        else:
            nslyr = int(ice_domain_size.nslyr)
            ice_init_column.init_snowtracers(
                ice_state_block("nt_smice", nslyr),
                ice_state_block("nt_smliq", nslyr),
                ice_state_block("nt_rhos", nslyr),
                ice_state_block("nt_rsnw", nslyr),
            )
    if tr_fsd:
        if runtype == "continue":
            ice_restart_column.restart_fsd = True
        if ice_restart_column.restart_fsd:
            ice_restart_column.read_restart_fsd()
        else:
            nfsd = int(ice_domain_size.nfsd)
            ice_init_column.init_fsd(ice_state_block("nt_fsd", nfsd))
    if tr_iso:
        if runtype == "continue":
            ice_restart_column.restart_iso = True
        if ice_restart_column.restart_iso:
            ice_restart_column.read_restart_iso()
        else:
            n_iso = int(icepack_tracers.n_iso)
            ice_init_column.init_isotope(
                ice_state_block("nt_isosno", n_iso),
                ice_state_block("nt_isoice", n_iso),
            )
    if tr_aero:  # ice aerosol
        if runtype == "continue":
            ice_restart_column.restart_aero = True
        if ice_restart_column.restart_aero:
            ice_restart_column.read_restart_aero()
        else:
            n_aero = int(icepack_tracers.n_aero)
            ice_init_column.init_aerosol(ice_state_block("nt_aero", 4 * n_aero))

    if runtype == "continue":
        if tr_brine:
            ice_restart_column.restart_hbrine = True
        if skl_bgc or z_tracers:
            ice_restart_column.restart_bgc = True

    if tr_brine or skl_bgc:  # brine height tracer
        ice_init_column.init_hbrine()
        if tr_brine and ice_restart_column.restart_hbrine:
            ice_restart_column.read_restart_hbrine()

    if skl_bgc or z_tracers:  # biogeochemistry
        if tr_fsd:
            ice_fileunits.nu_diag.write(
                "FSD implementation incomplete for use with BGC\n"
            )
        ice_init_column.init_bgc()

    #-----------------------------------------------------------------
    # aggregate tracers
    #-----------------------------------------------------------------
    tmask = ice_grid.tmask
    ny = ice_state.aice.shape[1]
    nx = ice_state.aice.shape[0]
    for j in range(ny):
        for i in range(nx):
            if tmask[i, j]:
                (
                    ice_state.aice[i, j],
                    ice_state.trcr[i, j, :],
                    ice_state.vice[i, j],
                    ice_state.vsno[i, j],
                    ice_state.aice0[i, j],
                ) = icepack_itd.icepack_aggregate(
                    aicen=ice_state.aicen[i, j, :],
                    trcrn=ice_state.trcrn[i, j, :, :],
                    vicen=ice_state.vicen[i, j, :],
                    vsnon=ice_state.vsnon[i, j, :],
                    trcr_depend=ice_state.trcr_depend,
                    trcr_base=ice_state.trcr_base,
                    n_trcr_strata=ice_state.n_trcr_strata,
                    nt_strata=ice_state.nt_strata,
                    Tf=ice_flux.Tf[i, j],
                )
            else:
                # tcraig, reset all tracer values on land to zero
                ice_state.trcrn[i, j, :, :] = c0
    ice_restart.final_restart()

def ice_state_slice(name):
    """One tracer, all categories. ``name`` is an ``nt_*`` attribute."""
    index = int(getattr(icepack_tracers, name))
    return ice_state.trcrn[:, :, index, :]


def ice_state_block(name, count):
    """``count`` consecutive tracers starting at ``nt_*``."""
    start = int(getattr(icepack_tracers, name))
    return ice_state.trcrn[:, :, start:start + count, :]

