import io
import logging
import numpy as np
import unittest

import legwork.source as source
from legwork._logging import LegworkFormatter, logger

from astropy import units as u


class FakeTTY(io.StringIO):
    """A stream that claims to be a terminal so that colours are used"""

    def isatty(self):
        return True


class Test(unittest.TestCase):
    """Tests that messages are logged (and formatted) correctly"""

    def test_logger_setup(self):
        """check that the LEGWORK logger is created with a handler attached"""
        self.assertTrue(logger.name == "LEGWORK")
        self.assertTrue(len(logger.handlers) > 0)

        # messages shouldn't be passed up to the root logger as well
        self.assertFalse(logger.propagate)

    def test_message_format(self):
        """check that each message is prefixed with the package name and level"""
        record = logging.LogRecord(name="LEGWORK", level=logging.WARNING, pathname="", lineno=0,
                                   msg="something is up", args=None, exc_info=None)

        # a stream that isn't a terminal shouldn't get any colour codes
        plain = LegworkFormatter(stream=io.StringIO()).format(record)
        self.assertTrue(plain == "LEGWORK warning: something is up")

        # but a terminal should get a bold, yellow prefix
        coloured = LegworkFormatter(stream=FakeTTY()).format(record)
        self.assertTrue(coloured == "\033[1m\033[33mLEGWORK warning\033[0m: something is up")

        # errors should be red instead
        record.levelno, record.levelname = logging.ERROR, "ERROR"
        coloured = LegworkFormatter(stream=FakeTTY()).format(record)
        self.assertTrue(coloured == "\033[1m\033[31mLEGWORK error\033[0m: something is up")

        # and info messages should only be bold
        record.levelno, record.levelname = logging.INFO, "INFO"
        coloured = LegworkFormatter(stream=FakeTTY()).format(record)
        self.assertTrue(coloured == "\033[1mLEGWORK info\033[0m: something is up")

    def test_source_warnings(self):
        """check that the Source class logs warnings rather than printing them"""
        n_values = 10
        m_1 = np.random.uniform(1, 10, n_values) * u.Msun
        m_2 = np.random.uniform(1, 10, n_values) * u.Msun
        dist = np.random.uniform(1, 30, n_values) * u.kpc
        f_orb = 10**(np.random.uniform(-5, -4, n_values)) * u.Hz
        ecc = np.zeros(n_values)

        # interpolating g for a small number of sources should warn the user
        with self.assertLogs("LEGWORK", level="WARNING") as log:
            source.Source(m_1=m_1, m_2=m_2, ecc=ecc, dist=dist, f_orb=f_orb,
                          interpolate_g=True, interpolate_sc=False)
        self.assertTrue("interpolate_g" in log.output[0])

    def test_source_info(self):
        """check that missing inclinations/polarisations are logged at the info level"""
        n_values = 10
        m_1 = np.random.uniform(1, 10, n_values) * u.Msun
        m_2 = np.random.uniform(1, 10, n_values) * u.Msun
        dist = np.random.uniform(1, 30, n_values) * u.kpc
        f_orb = 10**(np.random.uniform(-5, -4, n_values)) * u.Hz
        ecc = np.zeros(n_values)

        from astropy.coordinates import SkyCoord
        position = SkyCoord(np.random.uniform(0, 360, n_values) * u.deg,
                            np.arcsin(np.random.uniform(-1, 1, n_values)) * u.rad,
                            distance=dist, frame="galactic")

        with self.assertLogs("LEGWORK", level="INFO") as log:
            source.Source(m_1=m_1, m_2=m_2, ecc=ecc, dist=dist, f_orb=f_orb, position=position,
                          interpolate_g=False, interpolate_sc=False)
        self.assertTrue(any("inclinations" in message for message in log.output))
        self.assertTrue(any("polarisations" in message for message in log.output))

    def test_no_snr_error(self):
        """check that plotting without an SNR logs an error"""
        n_values = 10
        sources = source.Source(m_1=np.ones(n_values) * u.Msun, m_2=np.ones(n_values) * u.Msun,
                                ecc=np.zeros(n_values), dist=np.ones(n_values) * u.kpc,
                                f_orb=1e-4 * np.ones(n_values) * u.Hz,
                                interpolate_g=False, interpolate_sc=False)

        with self.assertLogs("LEGWORK", level="ERROR") as log:
            fig, ax = sources.plot_sources_on_sc()
        self.assertTrue("No SNR" in log.output[0])
        self.assertTrue(fig is None and ax is None)
