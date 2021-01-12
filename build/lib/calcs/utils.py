"""`utils` for gw calcs"""

from scipy.special import jv


def chirp_mass(m_1, m_2):
    """Computes chirp mass of a binary system

    Params
    ------
    m_1 : `float/array`
        more massive binary component in units of kg
    m_2 : `float/array`
        less massive binary component in units of kg

    Returns
    -------
    m_c : `float/array`
        chirp mass of the binary in units of kg
    """

    m_c = (m_1 * m_2)**(3/5) / (m_1 + m_2)**(1/5)
    return m_c

def peters_g(n, e):
    """Fourier decomposition of the gravitational wave signal
    from Peters and Matthews (1963)

    Params
    ------
    n : `int`
        harmonic of interest
    e : `array`
        eccentricity

    Returns
    -------
    g : `array`
        Fourier decomposition
    """

    bracket_1 = jv(n-2, n*e) - 2*e*jv(n-1, n*e) +\
                2/n*jv(n, n*e) + 2*e*jv(n+1, n*e) -\
                jv(n+2, n*e)
    bracket_2 = jv(n-2, n*e) - 2*jv(n, n*e) + jv(n+2, n*e)
    bracket_3 = jv(n, n*e)

    g = n**4/32 * (bracket_1**2 + (1-e**2) * bracket_2**2 +\
        4/(3*n**3) * bracket_3**2)

    return g
