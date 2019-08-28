GridFloat
=========

.. note::
   This class is only available if single precision support is enabled, in which case the marco SPFFT_SINGLE_PRECISION is defined in config.h.

.. note::
   A Grid object can be safely destroyed after transforms have been created. The transforms hold a reference counted objtect containing the allocated memory, which will remain valid until all transforms are destroyed as well.

.. doxygenclass:: spfft::GridFloat
   :project: SpFFT
   :members:
