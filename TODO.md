TODO
====

daostorm_3d
-----------

- implement "Z" model
- implement z fitting
- handle negative values
- improve non-uniform background handling
  (orig 50fd8fa5e939c140b25e33b901a7f40fbbd5e31f)
- implement low SNR finding improvements
  (orig b312853c7f30913b493519f30a5fb404bbf8df6c,
  66a22b242e4ded44920205286d3436f3545dda88)


Misc
----

- locator: Adjust threshold spin boxes to image data type range (e.g.
  0 - 1 for floating point, but with 3 decimals; 0-65535 for uint16, ...)
