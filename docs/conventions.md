# Notation and Conventions

* We use `u` to denote row coordinates and `v` to denote column coordinates. This means we index images using the `img[u, v]` format.
* When de-projecting an image to 3D space, we convert `u` values to `y` values and `v` values to `x` values. This means our frame of reference is the top left corner of the image with x pointing right and y pointing down.
* When scattering in matplotlib, we use `plt.scatter(vs, us)` to respect the above conventions.