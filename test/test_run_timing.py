"""Compilation interval accounting and log output, including cached JIT calls."""

import io
import re
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from cicecore.cicedyn.infrastructure.comm.serial import ice_timers
from cicecore.shared import ice_fileunits


class RunTimingTest(unittest.TestCase):
    def setUp(self):
        self.log = io.StringIO()
        self.output = patch.object(ice_fileunits, 'nu_diag', self.log)
        self.output.start()
        self.addCleanup(self.output.stop)
        self.addCleanup(setattr, ice_timers, '_run_timing', None)

    def test_nested_compilation_and_output_are_not_double_counted(self):
        timer = ice_timers.RunTiming()
        timer.started = 0.
        timer.intervals = [(0,4,'initialization'), (4,18,'run'), (18,20,'finalize'),
                           (1,3,'compile'), (2,2.5,'compile'), (6,8,'compile'),
                           (14,15,'compile'), (12,16,'history.write'),
                           (16,18,'restart'), (8,9,'profiling')]
        result = timer.totals(20.)
        self.assertEqual(result, {'compile':5., 'history.write':3., 'restart':2.,
                                 'profiling':1., 'initialization':2., 'run':5.,
                                 'finalize':2., 'other':0.})
        self.assertEqual(sum(result.values()), 20.)
        self.assertEqual(timer.segments[0][0],0.)
        self.assertEqual(timer.segments[-1][1],20.)
        for left,right in zip(timer.segments,timer.segments[1:]):
            self.assertEqual(left[1],right[0])
            self.assertNotEqual(left[2],right[2])
        self.assertEqual(sum(end-start for start,end,phase in timer.segments if phase=='run'),5.)

    def test_initial_output_recorded_without_step_timing(self):
        timer = ice_timers.RunTiming()
        with patch.object(ice_timers.time, 'perf_counter', return_value=5.):
            ice_timers.record_stage(None, 'history.write', 2.)
        self.assertEqual(timer.intervals, [(2.,5.,'history.write')])

    def test_real_jit_cached_execution_and_summary(self):
        values = jax.device_put(np.arange(128, dtype=np.float32))
        function = jax.jit(lambda x: jnp.sin(x * 0.731).sum())
        timer = ice_timers.RunTiming()
        with ice_timers.timing_phase('run'):
            jax.block_until_ready(function(values))
            self.assertGreater(timer.compile_events['trace'], 0)
            self.assertGreater(timer.compile_events['backend'], 0)
            counts = dict(timer.compile_events)
            jax.block_until_ready(function(values))
            self.assertEqual(timer.compile_events, counts)
        timer.finish(True)
        self.assertIsNone(ice_timers._run_timing)
        line = next(line for line in self.log.getvalue().splitlines()
                    if line.startswith('TIMING_TOTAL completed='))
        numbers = {name:float(value) for name,value in re.findall(r'(\w+_seconds)=([\d.]+)',line)}
        self.assertGreater(numbers['compile_seconds'],0)
        self.assertGreater(numbers['run_seconds'],0)
        self.assertAlmostEqual(numbers['wall_seconds'], sum(value for name,value in numbers.items()
                                                          if name != 'wall_seconds'), places=5)
        # A new session can reuse the executable without claiming compilation.
        timer = ice_timers.RunTiming()
        with ice_timers.timing_phase('run'):
            jax.block_until_ready(function(values))
        self.assertEqual(sum(timer.compile_events.values()),0)
        timer.finish(True)

    def test_failure_is_reported_and_listener_is_deactivated(self):
        from cicecore.drivers.standalone.cice import CICE
        with patch.object(CICE, 'CICE_Initialize', side_effect=ValueError('initialization failed')):
            with self.assertRaisesRegex(ValueError, 'initialization failed'):
                CICE.icemodel()
        self.assertIn('TIMING_TOTAL completed=false', self.log.getvalue())
        self.assertIsNone(ice_timers._run_timing)

    def test_unsupported_monitoring_does_not_claim_zero_compilation(self):
        with patch.object(ice_timers, '_compile_listeners_registered', False):
            with patch.object(ice_timers, 'monitoring', object()):
                timer = ice_timers.RunTiming()
                timer.finish(True)
        self.assertIn('compile_seconds=unavailable run_seconds=unavailable',self.log.getvalue())

    def test_fortran_table_keeps_unmeasured_and_disabled_distinct(self):
        ice_timers.init_ice_timers()
        ice_timers.configure_timer_scopes({
            'icepack_parameters':{'wave_spec':False, 'skl_bgc':False},
            'icepack_tracers':{'tr_fsd':False, 'tr_brine':False}})
        ice_timers.ice_timer_print_all()
        lines = [line for line in self.log.getvalue().splitlines() if re.match(r'Timer\s+\d+:', line)]
        self.assertEqual([int(line.split()[1][:-1]) for line in lines], list(range(1,21)))
        self.assertIn('N/A', lines[14])
        self.assertIn('0.00', lines[15])
        self.assertIn('id=16 measured=false disabled=true', self.log.getvalue())

    def test_unsampled_physics_adds_no_barrier_or_fictitious_time(self):
        ice_timers.init_ice_timers()
        timing = {'step':8, 'seconds':{}, 'detailed_physics':False}
        with patch.object(ice_timers, 'block_until_ready') as wait:
            ice_timers.record_stage(timing, 'physics.horizontal.transport', 0., object())
        wait.assert_not_called()
        self.assertEqual(ice_timers._timers[3].accum, 0.)
        self.assertFalse(ice_timers._timers[3].used)

    def test_stage_timers_report_sample_coverage_and_union_of_compilation(self):
        ice_timers.init_ice_timers()
        session = ice_timers.RunTiming()
        session.intervals = [(1.,3.,'compile'), (2.,2.5,'compile'), (4.,5.,'compile')]
        timing = {'step':7, 'seconds':{}, 'detailed_physics':True}
        with patch.object(ice_timers.time, 'perf_counter', return_value=6.):
            ice_timers.record_stage(timing, 'physics.therm1', 0., emit=False)
        ice_timers.ice_timer_print(6)
        self.assertIn('stage_steps=7', self.log.getvalue())
        self.assertIn('compile_seconds=3.000000 excluding_compile_seconds=3.000000', self.log.getvalue())
        self.assertEqual(ice_timers._timers[4].accum, 6.)
        self.assertEqual(ice_timers._timers[5].accum, 6.)


if __name__ == '__main__':
    unittest.main()
