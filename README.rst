Orbit Predictor
===============

.. image:: https://travis-ci.org/satellogic/orbit-predictor.svg?branch=master
    :target: https://travis-ci.org/satellogic/orbit-predictor
.. image:: https://coveralls.io/repos/github/satellogic/orbit-predictor/badge.svg?branch=master
    :target: https://coveralls.io/github/satellogic/orbit-predictor?branch=master


Orbit Predictor is a Python library to propagate orbits of Earth-orbiting objects (satellites, ISS, 
Santa Claus, etc) using `TLE (Two-Line Elements set) <https://en.wikipedia.org/wiki/Two-line_element_set>`_

All the hard work is done by Brandon Rhodes implementation of 
`SGP4 <https://github.com/brandon-rhodes/python-sgp4>`_. 

We can say *Orbit predictor* is kind of a "wrapper" for the python implementation of SGP4

To install it
-------------

You can install orbit-predictor from pypi::

    pip install orbit-predictor

Use example
-----------

When will be the ISS over Argentina?

:: 

    In [1]: from orbit_predictor.sources import EtcTLESource

    In [2]: from orbit_predictor.locations import ARG

    In [3]: source = EtcTLESource(filename="examples/iss.tle")

    In [4]: predictor = source.get_predictor("ISS")

    In [5]: predictor.get_next_pass(ARG)
    Out[5]: <PredictedPass ISS over ARG on 2017-11-10 22:48:10.607212>

    In [6]: predicted_pass = _

    In [7]: position = predictor.get_position(predicted_pass.aos)

    In [8]: ARG.is_visible(position)  # Can I see the ISS from this location?
    Out[8]: True

    In [9]: import datetime

    In [10]: position_delta = predictor.get_position(predicted_pass.los + datetime.timedelta(minutes=20))

    In [11]: ARG.is_visible(position_delta)
    Out[11]: False

    In [12]: tomorrow = datetime.datetime.utcnow() + datetime.timedelta(days=1)

    In [13]: predictor.get_next_pass(ARG, tomorrow, max_elevation_gt=20)
    Out[13]: <PredictedPass ISS over ARG on 2017-11-11 23:31:36.878827>


Simplified creation of predictor from TLE lines:

::

    In [1]: import datetime

    In [2]: from orbit_predictor.sources import get_predictor_from_tle_lines

    In [3]: TLE_LINES = (
                "1 43204U 18015K   18339.11168986  .00000941  00000-0  42148-4 0  9999",
                "2 43204  97.3719 104.7825 0016180 271.1347 174.4597 15.23621941 46156")

    In [4]: predictor = get_predictor_from_tle_lines(TLE_LINES)

    In [5]: predictor.get_position(datetime.datetime(2019, 1, 1))
    Out[5]: Position(when_utc=datetime.datetime(2019, 1, 1, 0, 0),
        position_ecef=(-5280.795613274576, -3977.487633239489, -2061.43227648734),
        velocity_ecef=(-2.4601788971676903, -0.47182217472755117, 7.167517631852518),
        error_estimate=None)

Currently you have available these sources
------------------------------------------

- Memorytlesource: in memory storage.
- EtcTLESource: a uniq TLE is stored in `/etc/latest_tle`
- WSTLESource: It reads a REST API currently used inside Satellogic. We are are working to make it publicly available.

How to contribute
-----------------

- Write pep8 complaint code. 
- Wrap the code on 100 collumns.
- Always use a branch for each feature and Merge Proposals.
- Always run the tests before to push. (test implies pep8 validation)
