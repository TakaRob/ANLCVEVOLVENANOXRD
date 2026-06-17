Find a robust and accurate algorithm that can detect the Bragg peaks (bright spots) in X-ray crystal diffraction (XRD) images.

1. Problem statement

An X-ray diffraction image of a crystal contains multiple bright spots or Bragg peaks. Different lattice planes deflect X-ray by different angles (2-theta). In an x-ray diffraction image, 2-theta is associated with the distance to the incident location of X-ray. When the sample contains crystals with different orientations, multiple Bragg peaks corresponding to a lattice plane can be found to lie on the arc associated with the 2-theta of that plane. 

Your task is to design an algorithm that, given an X-ray diffraction image, a map indicating the 2-theta at every pixel of that image, and a list of lattice planes to look for, identify and locate the Bragg peaks on the arc of those planes (we may use "plane" and "reflection" interchangebly here, so (111) plane and (111) reflection refer to the same thing). The Bragg peaks are bright spots that vary in intensity, contrast and size. The XRD images may contain background, which needs to be distinguished from Bragg peaks. Some Bragg peaks are faint and require careful design for the algorithm to distinguish them from the background.

2. Test data

The data folder contains the following:

- integrated_intensity_design.tiff: the XRD image where the Bragg peaks need to be identified and located.
- ttf.tiff: the 2-theta map.
- baseline.py: a baseline algorithm. Use this algorithm as a reference to understand the problem, and use its performance as a baseline. Find an algorithm that outperforms it.
- reflections.py: the 2-theta values and names of the valid lattice planes or reflections in the image. 
- labels.json: the ground truth Bragg peak locations (in pixels) in the image for every lattice plane.

Note: integrated_intensity_design.tiff has a few bad pixels and blocked regions where the values are negative. When visualizing the image, it is recommended to set the percentile range to 10 and 99. Adjust the bounds if needed. If your algorithm involves log-transform, be careful handling that.

3. Evaluation

Your algorithm should take the image, the 2-theta map, and a list of 2-theta values of the reflections. When evaluating an algorithm, you must find Bragg peaks on ALL the lattice planes in reflections.py. 

If a detected point is within 40 pixels to a ground truth point, consider that as a match. The goal is to capture as many ground truth points as possible while limiting false positives. Use a metric like f1 score.

When evaluating the baseline algorithm, use `line_tol=0.2`, and keep the default settings.

4. IO

The scripts of the algorithm you submit should take the input image (`--image`), two-theta file (`--two-theta`), reflection file (`--reflections`), output file path (`--output`) and other necessary flags. The output should be a csv file that contains three columns: `reflection`, `x`, `y`. Each row in this csv represents a detected spot. Additionally, if labels are provided, the script should calculate the metric value and print it.