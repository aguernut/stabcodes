from itertools import count

class MeasureClock:
    """
    Basic counter class to track the clock state of the quantum processor.
    This clock should incremented everytime a stim measurement is performed.

    Attributes
    ----------
    current: int
        The current time of the clock.

    Notes
    -----
    This is the under the hood implementation that guarantees coherent measurement in
    the :class:`StimExperiment` class.

    Examples
    --------
    >>> clock = MeasureClock()
    >>> index = next(clock)
    >>> index == clock.current == 0
    True
    >>> index = next(clock)
    >>> index == clock.current == 1
    True

    """

    def __init__(self):
        """Initiates a clock."""
        self._clock = count()
        self._current = 0

    def __next__(self):
        self._current = next(self._clock)
        return self._current

    @property
    def current(self):
        """Current time of the clock."""
        return self._current