TransformFloat
==============
.. note::
   This class is only available if single precision support is enabled, in which case the marco SPFFT_SINGLE_PRECISION is defined in config.h.

.. note::
   This class only holds an internal reference counted object. The object remains in a usable state even if the associated Grid object is destroyed. In addition, copying a transform only requires an internal copy of a shared pointer.


.. doxygenclass:: spfft::TransformFloat
   :project: SpFFT
   :members:
