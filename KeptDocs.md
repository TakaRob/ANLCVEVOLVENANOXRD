Xrd-app gui is a single gui with tabs
using PyQt5
Creating Project Directories


Raw Data (Formats?) 
We use beamline 19, where we calculate the coordinates.csv from interferometer data.




Binning
- What type, what were the pros and cons? show perfect row image
Peak finding (Labeling, CVEvolve, Algorithm Testing?)
 - figure out if it will work on diff scans
 - Runs a detector on each bin.
Shape Finding (what is this, methods, tolerance adjustments)
Rocking analysis for 3D



Claude.md and it's respective .mds will be instructions for agents using the CLI tools, (agent.md? we can have multiple) you have general access to tools linking, etc etc. whatever.
use this format in responese, you have 15 features ...
view using the gui. 


Single GUI (PyQt5, tabbed)
  Setup │ Programs │ View/Label │ Shape/Verify │ Device │ Orientation



  File Structure

  ```
Directory
| Project Folder
    metadata
    binned
    etc
    etc


└── xrd_app/
    ├── pyproject.toml          # entry point: xrd-app = xrd_app.cli:main
    ├── cli.py  config.py  app.py
    ├── tabs/                   # setup, programs, view_label, shape_verify,



    Data Loading

    - Creating a project.

    - Load single hdf5, will try to autodetect 

    - Loading from PONI, creating your own theta thing.
        can we make that easier



    Gui as Standalone
    - prompt to select file, based on stuff

    Guide for each gui


    Flesh out Save Algorithm better, what does it actually do?(1) sets the chosen
sensitivity, (2) calls the chosen noise-reduction module, (3) delegates to the
base algorithm — conforming to the **detector contract** (`precompute_tth`,
`detect_in_band(...)`) that `core/processing.py` already imports. Template in
`core/save_algorithm.py`.
