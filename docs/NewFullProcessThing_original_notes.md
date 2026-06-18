I want to create an easy to use python package where there is a single gui, 

each big button press that runs something basically calls a cli command, 
each of the indiviidual gui tools we had before will be combined into one

each page should have a help cli thing like app Setup --help with a simple text thing, the guis will show the math and algorithms used "general" and describe the visualizations


Pages

Setup:
Project Name

Load Data Directory (they select a scan) then it auto takes the upper directory
    "(Scan203) / (151hdf5 files)"

Load .PONI File, or other, 



Manual Popup: Can find radial histogram on #x# binned images, which is printed and saved as png with fully binned image, we will see fully binni jused image and histogram with 2theta overlay, this overlay is created
[ReflectionName] [Bragg Angle] [Width] 

[Create Button]


Programs:
Peak Finding Algorithm [SelectAlgorith] [SelectBinSize] [Run]
[status]

Shape Finding Algorithm [SelectAlgorithm] [SelectBinSize] [Select Saved Peaks/ Current peak finding algorithm ^]  [run]
[status]

*Want to scan all? use cli tool -all scans --algorithm.py --algorithm.py .... ... or just --all scans, then will be prompted to select from ...

[UseCVEvolve] popup
Pick Scan Directory
Pick Development Set
From there can open necessary .md files and add context files, 

View/Label

If data is labeled here, add an extra verified tag, like we have already? since it is useful for cvevolve
Our viewing gui, 
also need to add a button to save algorithm, (which is like what we had when we would run the algorithm with different sensitivity, and with the noise reduction on top) with a new name, or just a pythonfile that inserts the newsens and calls the noise reduction with a call to the base algorithm


Shape/Verification

Same new save alrogithm thing, and the same verification thing, we will want to be able to change the sensitivity of the shape finding as well, with the save feature.



Device View




Orientation Map





*We will want some modularity, so if someone wants to add another, it can be simple as linking their gui, then it will appear as a tab, 
*Also, these individual guis should be able to be loaded individually, so they can have like a setup page where the data is loaded, then the exact same stuff...




File Structure

APP
\SRC
\ShapeAlgorithms (all our algorithms are saved here) naming is manual or, if from CVevolve, Scan2035x5(desc)F1:##.#, or (SuperSpecialAlgorithm) will need a guide on how to put in your own, 
\PeakAlgorithms
\CombinedAlgorithms
\Projects
    \NamedProject (Created upon Project creation in GUI)
        \Raw (Simlink metadata, this file can have all the scan links in the same file)
        \Binned (if in WSL have that link metadata to avoid IO or just mount, so in this folder) 
            \Scan#
                \#x#Data...hdf5
        \Metadata (saves the poni, or other manual stuff from setup, also like cookies, will save the state of the gui, sensitivity, algorithm in the labeling tool, if we have selected anything)
        \Labels 
            \Scan203
                \algo1peaks.json
                \algo2peaks.json
                \algo2shapes.json
        \Figures (where .pngs are saved, histogram from setup, other that are saved etc)
        \CVEvolve if created in gui






Other: 

Algorithm Naming Conventions:



