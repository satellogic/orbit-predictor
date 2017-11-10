Orbit Predictor
===============

Orbit Predictor is a Python library to propagate orbits of Earth-orbiting objects (satellites, ISS, 
Santa Claus, etc) using `TLE (Two-Line Elements set) <https://en.wikipedia.org/wiki/Two-line_element_set>`_

Al the hard work is done by The Brandon Rhodes implementation of 
`SGP4 <https://github.com/brandon-rhodes/python-sgp4>`_. 

We can say *Orbit predictor* is kind of a "wrapper" for the python implementation of SGP4

To install it
-------------

You can install orbit-predictor from pypi::

    pip install orbit-predictor # WIP

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


`WSTLESource` needs the tle.satellogic.com service to be working. We are doing changes to have it public available.


Currently you have available these sources
------------------------------------------

- Memorytlesource: in memory storage.
- EtcTLESource: a uniq TLE is stored in `/etc/latest_tle`
- WSTLESource: It source is using the `TLE API. <http://tle.satellogics.com/api/tle/>`_


About HighAccuracyTLEPredictor 
------------------------------

The default 'predictor' code is tunned to low CPU usage. (IE: a Satellite computer). The 
error estimation is ~20 seconds. If you need more than that you can use the *HighAccuracyTLEPredictor*  
passing `precise=True` to `get_predictor()`. 


How to contribute
-----------------

- Write pep8 complaint code. 
- Wrap the code on 100 collumns.
- Always use a branch for each feature and Merge Proposals.
- Always run the tests before to push. (test implies pep8 validation)
