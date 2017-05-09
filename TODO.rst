TODO
====

daostorm_3d
-----------

- handle negative values
- Investigate default clamp improvements

  The default clamp has values in the order of magnitude of the
  expected fit values. This gets interesting (to say the least) when
  e. g. looking at z fitting, where depending on the unit (μm vs. nm)
  the clamp value has to be either 0.1 or 100…


Misc
----

- locator: Adjust threshold spin boxes to image data type range (e.g.
  0 - 1 for floating point, but with 3 decimals; 0-65535 for uint16, ...)
