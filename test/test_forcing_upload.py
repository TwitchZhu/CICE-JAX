"""Packed JRA55 inputs preserve fields and temperature-cap boundaries."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from cicecore.drivers.standalone.cice import CICE_RunMod as driver


class ForcingUploadTest(unittest.TestCase):
    def test_packed_fields_and_ice_cap(self):
        jax.config.update('jax_enable_x64', True)
        shape = (4, 3)
        fields = {name:np.full(shape, index + .25) for index, name in enumerate(driver._ATMOSPHERE_FIELDS)}
        fields['Tair'][:] = 280.
        fields['potT'][:] = 279.
        fields['Tair'][2, 1] = 265.
        area = jnp.asarray([[0., .1, .100001], [.5, .5, .5], [.5, .5, .5], [.5, .5, .5]])
        mask = jnp.ones(shape).at[1, 1].set(0.)
        clear = {'flat':jnp.ones(shape), 'fswthrun_ai':jnp.ones(shape + (5,))}
        bgc = {'Qa_iso':jnp.ones(shape + (3,))}
        # Previous device temperatures must not override the newly uploaded values.
        temperature = (jnp.full(shape, 999.), jnp.full(shape, 999.), area, mask, 273.15)
        for repeat in range(2):
            fields['fsw'][:] = repeat + .125
            packed = jnp.asarray(np.stack([fields[name] for name in driver._ATMOSPHERE_FIELDS]))
            actual, gases = jax.device_get(driver._forcing_prepare_compiled(clear, bgc, temperature, (1, 2, 1, 1), packed))
            for name in driver._ATMOSPHERE_FIELDS:
                if name not in ('Tair', 'potT'):
                    np.testing.assert_array_equal(actual[name], fields[name])
                    self.assertEqual(actual[name].dtype, fields[name].dtype)
            np.testing.assert_array_equal(actual['Tair'][0], [280., 280., 273.25])
            self.assertEqual(actual['Tair'][1, 1], 0.)
            self.assertEqual(actual['Tair'][2, 1], 265.)
            expected_potential = fields['potT'].copy()
            expected_potential[1:3, 1:2] = actual['Tair'][1:3, 1:2]
            np.testing.assert_array_equal(actual['potT'], expected_potential)
            for name in clear:
                np.testing.assert_array_equal(actual[name], np.zeros(clear[name].shape))
            np.testing.assert_array_equal(gases['Qa_iso'], np.zeros(shape + (3,)))

    def test_unpacked_and_no_temperature_paths(self):
        jax.config.update('jax_enable_x64', True)
        clear = {'flat':jnp.ones((2, 2))}
        flux, bgc = jax.device_get(driver._forcing_prepare_compiled(clear, {}, None, (0, 1, 0, 1)))
        self.assertEqual(set(flux), {'flat'})
        self.assertEqual(bgc, {})
        np.testing.assert_array_equal(flux['flat'], np.zeros((2, 2)))
        temperature = (jnp.full((2, 2), 280.), jnp.full((2, 2), 279.),
                       jnp.ones((2, 2)), jnp.ones((2, 2)), 273.15)
        flux, _ = jax.device_get(driver._forcing_prepare_compiled(clear, {}, temperature, (0, 1, 0, 1)))
        np.testing.assert_array_equal(flux['Tair'], np.full((2, 2), 273.25))
        np.testing.assert_array_equal(flux['potT'], np.full((2, 2), 273.25))


if __name__ == '__main__':
    unittest.main()
