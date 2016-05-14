TODO
====

daostorm_3d
-----------

- handle negative values
- implement low SNR finding improvements
  (orig b312853c7f30913b493519f30a5fb404bbf8df6c,
  66a22b242e4ded44920205286d3436f3545dda88)
- Investigate default clamp improvements
  
  The default clamp has values in the order of magnitude of the
  expected fit values. This gets interesting (to say the least) when
  e. g. looking at z fitting, where depending on the unit (μm vs. nm)
  the clamp value has to be either 0.1 or 100…


Misc
----

- locator: Adjust threshold spin boxes to image data type range (e.g.
  0 - 1 for floating point, but with 3 decimals; 0-65535 for uint16, ...)
