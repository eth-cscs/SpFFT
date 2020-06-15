GridFloat
=========
.. note::
   A Grid handle can be safely destroyed after Transform handles have been created, since internal reference counting used to prevent the release of resources while still in use.

.. note::
   These functions are only available if single precision support is enabled, in which case the marco SPFFT_SINGLE_PRECISION is defined in config.h.

.. doxygenfile:: spfft/grid_float.h
   :project: SpFFT

