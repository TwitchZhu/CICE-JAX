"""Main driver routine for CICE.  Initializes and steps through the model.
This program should be compiled if CICE is run as a separate executable,
 but not if CICE subroutines are called from another program (e.g., CAM).

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

2008: E. Hunke moved ESMF code to its own driver
2006: Converted to free form source (F90) by Elizabeth Hunke
authors Elizabeth C. Hunke and William H. Lipscomb, LANL
"""

from cicecore.drivers.standalone.cice.CICE_InitMod import CICE_Initialize
from cicecore.drivers.standalone.cice.CICE_FinalMod import CICE_Finalize
from cicecore.drivers.standalone.cice.CICE_JAXState import pack_initial_state
from cicecore.drivers.standalone.cice.CICE_RunMod import CICE_Run


def icemodel(wave_seed=None, diagnose=False, profile_dir=None, profile_steps=(5, 7),
             detailed_physics_steps=None):
    #-----------------------------------------------------------------
    # Initialize CICE
    #-----------------------------------------------------------------

    CICE_Initialize()

    #-----------------------------------------------------------------
    # Run CICE
    #-----------------------------------------------------------------

    pytree, static = pack_initial_state(wave_seed=wave_seed)
    pytree = CICE_Run(pytree, static, diagnose=diagnose,
                      profile_dir=profile_dir, profile_steps=profile_steps,
                      detailed_physics_steps=detailed_physics_steps)

    #-----------------------------------------------------------------
    # Finalize CICE
    #-----------------------------------------------------------------

    CICE_Finalize()


if __name__ == "__main__":
    icemodel()
