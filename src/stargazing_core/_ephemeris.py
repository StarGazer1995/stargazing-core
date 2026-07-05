"""One-time astropy ephemeris configuration.

Importing this module sets the solar-system ephemeris to ``'builtin'`` so
that planet and moon calculations work without network access.  Must be
imported before any other astropy coordinate transformations.
"""

from astropy.coordinates import solar_system_ephemeris

solar_system_ephemeris.set('builtin')
