"""JRA55 record reuse, year transitions and file lifetime."""

import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import netCDF4
import numpy as np

from cicecore.cicedyn.general import ice_forcing as forcing
from cicecore.drivers.standalone.cice import CICE


class ForcingFileTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        names = ('airtmp', 'wndewd', 'wndnwd', 'spchmd', 'glbrad', 'dlwsfc', 'ttlpcp')
        for year in (2004, 2005, 2006):
            with netCDF4.Dataset(Path(self.directory.name) / f'forcing_{year}.nc', 'w') as ds:
                for name, size in (('time', 16), ('nj', 2), ('ni', 3)):
                    ds.createDimension(name, size)
                for index, name in enumerate(names):
                    variable = ds.createVariable(name, 'f8', ('time', 'nj', 'ni'))
                    variable[:] = (year * 100 + index * 20 + np.arange(1, 17))[:, None, None]
        settings = [
            (forcing, 'uwind_file', str(Path(self.directory.name) / 'forcing_2005.nc')),
            (forcing, 'fyear', 2005), (forcing, 'fyear_final', 2006),
            (forcing, 'frec_info', np.full((2, 2), -99)),
            (forcing, 'local_debug', False), (forcing, 'debug_forcing', False),
            (forcing.ice_calendar, 'days_per_year', 2),
            (forcing.ice_calendar, 'yday', 1), (forcing.ice_calendar, 'msec', 0),
            (forcing.ice_init, 'atm_data_type', 'JRA55'),
            (forcing.ice_init, 'fyear_init', 2004),
            (forcing.ice_domain_size, 'nx_global', 3),
            (forcing.ice_domain_size, 'ny_global', 2),
            (forcing.ice_grid, 'hm', np.ones((3, 2))),
            (forcing.ice_fileunits, 'nu_diag', io.StringIO()),
        ]
        for name in ('Tair', 'uatm', 'vatm', 'Qa', 'fsw', 'flw', 'fsnow'):
            settings.extend(((forcing, name + '_data', np.zeros((3, 2, 2))),
                             (forcing.ice_flux, name, np.zeros((3, 2)))))
        for module, name, value in settings:
            setting = patch.object(module, name, value)
            setting.start()
            self.addCleanup(setting.stop)
        for setting in (
            patch.object(forcing, '_params', return_value={'Tffresh':273.15, 'secday':86400.}),
            patch.object(forcing, 'scatter_global', side_effect=lambda dest, src, *args: np.copyto(dest, src)),
        ):
            setting.start()
            self.addCleanup(setting.stop)
        forcing.close_forcing()
        self.addCleanup(forcing.close_forcing)

    def test_reuses_file_and_interpolates_new_records(self):
        with patch.object(forcing, '_nc_open', wraps=forcing._nc_open) as opened:
            for seconds in (0, 3600, 10800, 14400, 21600):
                forcing.ice_calendar.msec = seconds
                forcing.JRA55_data(defer_ice_correction=True)
                np.testing.assert_allclose(forcing.ice_flux.Tair, 200501 + seconds / 10800)
                np.testing.assert_array_equal(forcing.ice_flux.fsw, 200581 + seconds // 10800)
            self.assertEqual(opened.call_count, 1)
            handle = forcing._jra_dataset
            self.assertTrue(handle.isopen())
            forcing.close_forcing()
            forcing.close_forcing()
            self.assertFalse(handle.isopen())

    def test_year_transition_and_forcing_cycle(self):
        for final_year, next_year in ((2006, 2006), (2005, 2004)):
            with self.subTest(next_year=next_year):
                forcing.close_forcing()
                forcing.frec_info[:] = -99
                forcing.fyear = 2005
                forcing.fyear_final = final_year
                forcing.ice_calendar.yday = 2
                forcing.ice_calendar.msec = 64800
                forcing.JRA55_data(defer_ice_correction=True)
                previous = forcing._jra_dataset
                forcing.ice_calendar.msec = 75600
                forcing.JRA55_data(defer_ice_correction=True)
                self.assertFalse(previous.isopen())
                np.testing.assert_array_equal(forcing.Tair_data[:, :, 0], 200516)
                np.testing.assert_array_equal(forcing.Tair_data[:, :, 1], next_year * 100 + 1)
                self.assertTrue(forcing.uwind_file.endswith('2005.nc'))
                retained = forcing._jra_dataset
                forcing.fyear = next_year
                forcing.ice_calendar.yday = 1
                forcing.ice_calendar.msec = 0
                forcing.JRA55_data(defer_ice_correction=True)
                self.assertIs(forcing._jra_dataset, retained)
                np.testing.assert_array_equal(forcing.ice_flux.Tair, next_year * 100 + 1)

    def test_read_failure_releases_handle_and_restores_path(self):
        forcing.JRA55_data(defer_ice_correction=True)
        handle = forcing._jra_dataset
        forcing.ice_calendar.msec = 10800
        with patch.object(forcing, '_jra_read_slot', side_effect=OSError('read failed')):
            with self.assertRaisesRegex(OSError, 'read failed'):
                forcing.JRA55_data(defer_ice_correction=True)
        self.assertFalse(handle.isopen())
        self.assertIsNone(forcing._jra_dataset)
        self.assertIsNone(forcing._jra_dataset_path)
        self.assertTrue(forcing.uwind_file.endswith('2005.nc'))

    def test_open_failure_closes_previous_file(self):
        forcing.ice_calendar.yday = 2
        forcing.ice_calendar.msec = 64800
        forcing.JRA55_data(defer_ice_correction=True)
        handle = forcing._jra_dataset
        forcing.ice_calendar.msec = 75600
        with patch.object(forcing, '_nc_open', side_effect=RuntimeError('open failed')):
            with self.assertRaisesRegex(RuntimeError, 'open failed'):
                forcing.JRA55_data(defer_ice_correction=True)
        self.assertFalse(handle.isopen())
        self.assertIsNone(forcing._jra_dataset_path)
        self.assertTrue(forcing.uwind_file.endswith('2005.nc'))

    def test_model_exit_releases_file_on_success_and_failure(self):
        for failing_phase in (None, 'CICE_Initialize', 'CICE_Run'):
            with self.subTest(failing_phase=failing_phase):
                forcing.frec_info[:] = -99
                forcing.JRA55_data(defer_ice_correction=True)
                handle = forcing._jra_dataset
                with patch.object(CICE, 'CICE_Initialize') as initialize, \
                     patch.object(CICE, 'pack_initial_state', return_value=({}, {})), \
                     patch.object(CICE, 'CICE_Run', return_value={}) as run, \
                     patch.object(CICE, 'CICE_Finalize'):
                    if failing_phase:
                        (initialize if failing_phase == 'CICE_Initialize' else run).side_effect = RuntimeError('model failed')
                        with self.assertRaisesRegex(RuntimeError, 'model failed'):
                            CICE.icemodel()
                    else:
                        CICE.icemodel()
                self.assertFalse(handle.isopen())
                self.assertIsNone(forcing._jra_dataset)


if __name__ == '__main__':
    unittest.main()
