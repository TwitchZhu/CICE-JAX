"""Packed diagnostic transfers preserve report structure, shapes and dtypes."""

from collections import OrderedDict
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from cicecore.cicedyn.analysis import ice_diagnostics as diagnostics


class PackedDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        jax.config.update('jax_enable_x64', True)
        self.static = {name:{} for name in (
            'ice_domain_size','ice_domain','icepack_tracers',
            'icepack_parameters','ice_dyn_shared')}
        for name, value in dict(npnt=0, print_global=True, print_points=True, aice_extmin=.15).items():
            setting = patch.object(diagnostics,name,value)
            setting.start()
            self.addCleanup(setting.stop)

    def test_mixed_types_and_changing_inputs(self):
        state = {'pairs':jnp.array([[1.,-0.],[np.inf,np.nan]],dtype=jnp.float64),
                 'scalar':jnp.array(3.5,dtype=jnp.float32),
                 'count':jnp.array(2**60+123,dtype=jnp.int64),
                 'mask':jnp.array([True,False]),'empty':jnp.empty((0,2))}
        def report(state, dt, config):
            return {'summary':OrderedDict([('z',state['pairs']),('a',state['scalar'])]),
                    'points':[{'count':state['count'],'mask':state['mask']},{}],
                    'empty':state['empty'],'disabled':[]}
        with patch.object(diagnostics,'_runtime_diags_values',report):
            values,unpack = diagnostics.make_runtime_diags(self.static,state,1.)
            for repeat in range(2):
                state = dict(state,count=state['count']+repeat,scalar=state['scalar']+repeat)
                buffers = jax.device_get(values(state,1.))
                self.assertEqual(len(buffers),4)
                expected,structure = jax.tree_util.tree_flatten(jax.device_get(report(state,1.,{})))
                actual,actual_structure = jax.tree_util.tree_flatten(unpack(buffers))
                self.assertEqual(structure,actual_structure)
                for a,b in zip(expected,actual):
                    self.assertEqual(a.dtype,b.dtype)
                    self.assertEqual(a.shape,b.shape)
                    np.testing.assert_array_equal(a,b)
                    if a.dtype.kind == 'f':
                        np.testing.assert_array_equal(np.signbit(a),np.signbit(b))

    def test_empty_report(self):
        with patch.object(diagnostics,'_runtime_diags_values',return_value={'points':[],'summary':{}}):
            values,unpack = diagnostics.make_runtime_diags(self.static,{},1.)
            self.assertEqual(unpack(jax.device_get(values({},1.))),{'points':[],'summary':{}})


if __name__ == '__main__':
    unittest.main()
