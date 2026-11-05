"""Transfer the completed NumPy initialization state to one JAX device.

The module groups arrays by their Fortran module names so that the time-step
translation can refer to the same fields. File handles, forcing file caches,
history writers and timers remain on the host. Nothing in this module runs
the time loop or changes the NumPy initialization arrays.
"""

import numpy as np

import jax

from cicecore.cicedyn.dynamics import ice_dyn_eap
from cicecore.cicedyn.dynamics import ice_dyn_evp
from cicecore.cicedyn.dynamics import ice_dyn_shared
from cicecore.cicedyn.dynamics import ice_dyn_vp
from cicecore.cicedyn.dynamics import ice_transport_driver
from cicecore.cicedyn.dynamics import ice_transport_remap
from cicecore.cicedyn.analysis import ice_diagnostics
from cicecore.cicedyn.general import ice_flux
from cicecore.cicedyn.general import ice_flux_bgc
from cicecore.cicedyn.general import ice_init
from cicecore.cicedyn.general import ice_state
from cicecore.cicedyn.general import ice_step_mod
from cicecore.cicedyn.infrastructure import ice_domain
from cicecore.cicedyn.infrastructure import ice_grid
from cicecore.cicedyn.infrastructure import ice_restoring
from cicecore.cicedyn.infrastructure.comm.serial import ice_boundary
from cicecore.shared import ice_arrays_column
from cicecore.shared import ice_calendar
from cicecore.shared import ice_domain_size
from cicecore.shared import ice_init_column
from icepack.columnphysics import icepack_parameters
from icepack.columnphysics import icepack_fsd
from icepack.columnphysics import icepack_orbital
from icepack.columnphysics import icepack_shortwave_data
from icepack.columnphysics import icepack_therm_shared
from icepack.columnphysics import icepack_tracers


jax.config.update("jax_enable_x64", True)


def _numeric_arrays(module):
    """Collect numeric module arrays; this is shared by all state groups."""
    arrays = {}
    for name, value in vars(module).items():
        if name.startswith("_") or not isinstance(value, np.ndarray):
            continue
        if value.dtype.kind in "biufc":
            arrays[name] = value
    return arrays


def pack_initial_state(wave_seed=None):
    """Return the initial device pytree and host-only static configuration.

    Floating-point leaves in ``state`` and ``parameters`` may later be chosen
    as CNOP controls. This function does not choose a particular perturbation
    variable and does not differentiate through NumPy initialization.
    """
    if ice_state.aicen is None or ice_flux.sst is None:
        raise RuntimeError("(pack_initial_state) call CICE_Initialize first")

    state_modules = (
        ("ice_state", ice_state),
        ("ice_flux", ice_flux),
        ("ice_flux_bgc", ice_flux_bgc),
        ("ice_arrays_column", ice_arrays_column),
        ("ice_restoring", ice_restoring),
        ("ice_dyn_shared", ice_dyn_shared),
        ("ice_dyn_evp", ice_dyn_evp),
        ("ice_dyn_eap", ice_dyn_eap),
        ("ice_dyn_vp", ice_dyn_vp),
        ("ice_transport_driver", ice_transport_driver),
        ("ice_transport_remap", ice_transport_remap),
        ("icepack_therm_shared", icepack_therm_shared),
    )
    constant_modules = (
        ("ice_grid", ice_grid),
        ("ice_boundary", ice_boundary),
        ("icepack_shortwave_data", icepack_shortwave_data),
        ("icepack_tracers", icepack_tracers),
        ("icepack_fsd", icepack_fsd),
    )

    host_state = {name: _numeric_arrays(module) for name, module in state_modules}
    if (bool(icepack_tracers.tr_fsd) and bool(icepack_parameters.wave_spec)
            and str(icepack_parameters.wave_spec_type).strip() == "random"):
        if wave_seed is None:
            raise ValueError("random wave fracture requires an explicit wave_seed")
        host_state["wave_rng"] = {"key": jax.random.key(int(wave_seed))}
    # Fortran keeps these budget scalars as module variables. They are first
    # assigned by init_mass_diags at the beginning of the first time step.
    host_state["ice_diagnostics"] = {
        name: np.asarray(0.0, dtype=np.float64)
        for name in ("totmn", "totms", "totmin", "totmis", "totsn", "totss", "toten", "totes")
    }
    host_state["ice_diagnostics"].update(_numeric_arrays(ice_diagnostics))
    host_state["ice_step_mod"] = {
        "uvelT_icep": np.zeros_like(ice_state.uvel),
        "vvelT_icep": np.zeros_like(ice_state.vvel),
        "snow_mass_error": np.zeros_like(ice_state.aice, dtype=np.bool_),
        "snow_energy_error": np.zeros_like(ice_state.aice, dtype=np.bool_),
        "therm1_failed": np.zeros_like(ice_state.aice, dtype=np.bool_),
        "therm1_aerosol_warning": np.zeros_like(ice_state.aice, dtype=np.bool_),
        "bgc_failed": np.zeros_like(ice_state.aice, dtype=np.bool_),
        "therm2_failed": np.zeros_like(ice_state.aice, dtype=np.bool_),
        "remap_failed": np.zeros_like(ice_state.aice, dtype=np.bool_),
        "remap_conservation_failed": np.bool_(False),
        "therm2_itd_failed": np.zeros_like(ice_state.aice, dtype=np.bool_),
        "therm2_new_ice_failed": np.zeros_like(ice_state.aice, dtype=np.bool_),
        "therm2_weld_failed": np.zeros_like(ice_state.aice, dtype=np.bool_),
        "therm2_cleanup_failed": np.zeros_like(ice_state.aice, dtype=np.bool_),
        "therm2_cleanup_initial_area_failed": np.zeros_like(ice_state.aice, dtype=np.bool_),
        "therm2_cleanup_rebin_failed": np.zeros_like(ice_state.aice, dtype=np.bool_),
        "therm2_cleanup_negative_area_failed": np.zeros_like(ice_state.aice, dtype=np.bool_),
        "therm2_cleanup_excess_area_failed": np.zeros_like(ice_state.aice, dtype=np.bool_),
        "therm2_area_entry": np.zeros_like(ice_state.aice),
        "therm2_area_after_itd": np.zeros_like(ice_state.aice),
        "therm2_area_after_new_ice": np.zeros_like(ice_state.aice),
        "therm2_area_after_lateral_melt": np.zeros_like(ice_state.aice),
        "therm2_area_after_cleanup": np.zeros_like(ice_state.aice),
        "wave_failed": np.zeros_like(ice_state.aice, dtype=np.bool_),
        "ridge_failed": np.zeros_like(ice_state.aice, dtype=np.bool_),
    }
    for name, length in (
        ("pdhi", ice_diagnostics.npnt), ("pdhs", ice_diagnostics.npnt),
        ("pde", ice_diagnostics.npnt),
        ("totison", icepack_tracers.max_iso), ("totisos", icepack_tracers.max_iso),
        ("totaeron", icepack_tracers.max_aero), ("totaeros", icepack_tracers.max_aero),
    ):
        host_state["ice_diagnostics"].setdefault(name, np.zeros(int(length), dtype=np.float64))
    constants = {name: _numeric_arrays(module) for name, module in constant_modules}
    constants["icepack_orbital"] = {
        name: np.asarray(getattr(icepack_orbital, name), dtype=np.float64)
        for name in ("eccen", "mvelpp", "lambm0", "obliqr")
    }
    constants["ice_init_column"] = {
        "zbgc_init_frac": np.asarray(
            ice_init_column.zbgc_init_frac
            if ice_init_column.zbgc_init_frac is not None
            else np.zeros(0, dtype=np.float64)),
        "R_S2N": np.asarray(
            ice_init_column.R_S2N
            if ice_init_column.R_S2N is not None
            else np.zeros(0, dtype=np.float64)),
    }
    for name in ("zbgc_frac_init", "bgc_tracer_type", "tau_ret", "tau_rel"):
        value = getattr(ice_init_column, name)
        constants["ice_init_column"][name] = np.asarray(
            value if value is not None else np.zeros(0, dtype=np.float64))
    constants["ice_init_column"]["kscavz"] = np.asarray(
        [0.03] * 23 + [0.20, 0.02, 0.02, 0.01, 0.01, 0.03],
        dtype=np.float64)
    iglob, jglob = ice_boundary.global_indices()
    constants["ice_boundary"].update({
        "iglob": np.asarray(iglob),
        "jglob": np.asarray(jglob),
    })

    # Keep the initialized physical scalars on the device as floating leaves.
    # A time-step routine must use these leaves rather than the module globals
    # whenever a parameter is chosen as a differentiable control.
    parameters = {}
    for module_name, module in (
        ("icepack_parameters", icepack_parameters),
        ("ice_dyn_shared", ice_dyn_shared),
        ("icepack_therm_shared", icepack_therm_shared),
        ("ice_dyn_eap", ice_dyn_eap),
        ("ice_dyn_vp", ice_dyn_vp),
    ):
        values = _numeric_arrays(module) if module_name == "icepack_parameters" else {}
        for name, value in vars(module).items():
            if name.startswith("_") or isinstance(value, (bool, np.bool_)):
                continue
            if isinstance(value, (float, np.floating)):
                values[name] = np.asarray(value, dtype=np.float64)
        parameters[module_name] = values

    # The namelist scalars above are the differentiable parameter leaves.
    # The fixed coefficients below have no scalar namelist counterpart.
    bgc_parameters = {
        "graze_exponent": np.asarray([0.333, 1.0, 1.0], dtype=np.float64),
        "graze_conc": np.asarray(1.36, dtype=np.float64),
        "phi_sk": np.asarray(0.30, dtype=np.float64),
        "R_gC2molC": np.asarray(12.01, dtype=np.float64),
    }
    parameters["icepack_zbgc_shared"] = bgc_parameters

    # Namelist switches and dimensions select JIT branches and array shapes.
    # They are ordinary host values, separate from differentiable leaves.
    static = {}
    for module_name, module in (
        ("ice_init", ice_init),
        ("ice_calendar", ice_calendar),
        ("ice_domain_size", ice_domain_size),
        ("ice_domain", ice_domain),
        ("ice_grid", ice_grid),
        ("ice_flux", ice_flux),
        ("ice_arrays_column", ice_arrays_column),
        ("icepack_parameters", icepack_parameters),
        ("icepack_tracers", icepack_tracers),
        ("ice_dyn_shared", ice_dyn_shared),
        ("icepack_therm_shared", icepack_therm_shared),
        ("ice_dyn_vp", ice_dyn_vp),
    ):
        static[module_name] = {
            name: value.item() if isinstance(value, np.generic) else value
            for name, value in vars(module).items()
            if not name.startswith("_")
            and isinstance(value, (bool, int, str, np.bool_, np.integer))
        }
    static["ice_calendar"]["dt"] = float(ice_calendar.dt)
    static["ice_calendar"]["dt_dyn"] = float(ice_calendar.dt_dyn)
    static["ice_calendar"]["yday"] = float(ice_calendar.yday)
    static["ice_init"]["trestore"] = float(ice_init.trestore)
    for name in ("debug_model_ponds_i", "debug_model_ponds_j"):
        static["ice_init"][name] = tuple(int(value) for value in getattr(ice_init, name))
    # The Fortran ibc padding walk uses only the fixed grid layout.
    static["ice_domain"]["east_edge_limit"] = int(len(iglob) - 1 - np.count_nonzero(
        (np.asarray(iglob) == 0) & (np.sum(jglob) != 0)
    ))
    static["ice_domain"]["north_edge_limit"] = int(len(jglob) - 1 - np.count_nonzero(
        (np.asarray(jglob) == 0) & (np.sum(iglob) != 0)
    ))
    static["ice_domain_size"]["physical_bounds"] = tuple(
        int(index) for index in ice_domain_size.physical_bounds()
    )
    # Tracer layout is fixed after initialization. Keep the loop indices on
    # the host so JAX tracing never converts device arrays to Python ints.
    static["ice_state_tracers"] = {
        "trcr_depend": tuple(int(value) for value in ice_state.trcr_depend),
        "n_trcr_strata": tuple(int(value) for value in ice_state.n_trcr_strata),
        "nt_strata": tuple(tuple(int(value) for value in row) for row in ice_state.nt_strata),
        "trcr_base": tuple(tuple(float(value) for value in row) for row in ice_state.trcr_base),
    }
    for name in (
        "bio_index", "bio_index_o", "nlt_zaero", "nt_zaero",
        "nt_bgc_N", "nlt_bgc_N", "nt_bgc_chl", "nlt_bgc_chl",
        "nt_bgc_DOC", "nlt_bgc_DOC", "nt_bgc_DON", "nlt_bgc_DON",
        "nt_bgc_DIC", "nlt_bgc_DIC", "nt_bgc_Fed", "nlt_bgc_Fed",
        "nt_bgc_Fep", "nlt_bgc_Fep",
    ):
        values = getattr(icepack_tracers, name)
        if values is not None:
            static["icepack_tracers"][name] = tuple(int(value) for value in values)

    pytree = jax.device_put({
        "state": host_state,
        "constants": constants,
        "parameters": parameters,
    })
    return pytree, static
