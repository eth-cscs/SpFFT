GridFloat
=========

.. note::
   This class is only available if single precision support is enabled, in which case the marco SPFFT_SINGLE_PRECISION is defined in config.h.

.. note::
   A Grid object can be safely destroyed after Transform objects have been created, since internal reference counting used to prevent the release of resources while still in use.


.. doxygenclass:: spfft::GridFloat
   :project: SpFFT
   :members:
