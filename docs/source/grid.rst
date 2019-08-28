Grid
====
.. note::
   A Grid object can be safely destroyed after transforms have been created. The transforms hold a reference counted objtect containing the allocated memory, which will remain valid until all transforms are destroyed as well.


.. doxygenclass:: spfft::Grid
   :project: SpFFT
   :members:
