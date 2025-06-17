3D Skeletonization and Topology Extraction Pipeline
The data is placed in the DATA folder
The code is placed in the Python folder
Parameters and thresholds can be modified in the code


This project implements a complete workflow for 3D skeleton extraction, topology generation, and visualization from gridded property data.

Propagation
Propagates from a seed point across the 3D grid based on property values.

Skeleton
Extracts skeleton points and segments from the propagation result using a density threshold.

Topology
Builds the branch structure and hierarchy from skeleton segments.

Visualization
Reads generated skeleton files and visualizes 3D skeleton structures with branch indices.



Core Classes & Functions
Propagation	Handles property propagation from a seed cell through neighboring cells.
Skeleton	Extracts skeleton nodes and edges based on density thresholding.
Edge	Represents an edge (connection between two nodes) in the skeleton.
Branch	Represents a branch (ordered list of connected points).
Topology	Builds branches and merges edges into hierarchical structures.
read_skeleton_file()	Reads skeleton files generated after skeletonization and topology processing.
visualize_skeleton()	3D visualization of the skeleton with branch indices.

Data interpretation
1,0,0,0 Node index 1, coordinate(0,0,0)
2,0,0,0 Node Index 2, Coordinate(0,0,1)
3,1,0,0 node index 3,coordinate(1,0,0)
4,1,0,1 Node index4,Coordinate(1,0,1)
end of skeleton
segment
1,2,2,1 Node 1 is connected to Node 2 with topology level 2 and branch number 1
1,3,1,3 Node 3 is connected to node 1 with topology level 1 and branch number 3
3,4,1,3 Node 1 is connected to Node 4 with topology level 1 and branch number 3
