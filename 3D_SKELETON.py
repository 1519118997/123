import os
import heapq
import numpy as np
import heapq
import matplotlib.pyplot as plt
import numpy as np
###################################################
###################################################
# Class representing a cell in the 3D grid.
# Each cell has an index and a value (slowness),
# and supports comparison based on its value for priority queue operations.
class Cell:
    def __init__(self, index=0, value=0.0):
        self.index = index  # Linear index of the cell in 3D grid
        self.value = -value  # Negate the value to convert min-heap to max-heap

    def __lt__(self, other):
        # Define less-than for heap operations (min-heap behavior)
        return self.value < other.value

###################################################
# Class that performs propagation in a 3D grid.
# It expands from a seed point, visits neighboring cells based on slowness,
# and records parent-child relationships for density calculation.
class Propagation:
    def __init__(self, prop, nullV, ni, nj, nk, seed_i, seed_j, seed_k):
        self.nullValue = nullV  # Value representing invalid or missing data
        self.seed_i = seed_i
        self.seed_j = seed_j
        self.seed_k = seed_k
        self.nI = ni  # Grid size along I (x-axis)
        self.nJ = nj  # Grid size along J (y-axis)
        self.nK = nk  # Grid size along K (z-axis)
        self.parentIndex = [-1] * (ni * nj * nk)  # Store parent of each cell
        self.slowness = prop  # Slowness values for each cell
        self.narrowBand = []  # Priority queue for active front cells
        self.endCells = []  # Store end cells (leaf nodes)

        # Initialize with seed point
        cell = Cell()
        cell.index = self.coord1d(seed_i, seed_j, seed_k)
        self.parentIndex[cell.index] = ni * nj * nk  # Mark seed parent with special value
        heapq.heappush(self.narrowBand, Cell(cell.index, self.slowness[cell.index]))

    # Convert linear index to 3D coordinates
    def coord3d(self, index):
        out = [0, 0, 0]
        out[0] = index // (self.nJ * self.nK)
        out[1] = (index % (self.nJ * self.nK)) // self.nK
        out[2] = index % self.nK
        return out
    
    # Convert 3D coordinates to linear index
    def coord1d(self, i, j, k):
        return i * self.nJ * self.nK + j * self.nK + k

    # Check whether coordinates are within grid boundaries
    def coord_exists(self, i, j, k):
        return 0 <= i < self.nI and 0 <= j < self.nJ and 0 <= k < self.nK

    # Get unvisited valid neighbors of a given cell
    def neighbors(self, position1D):
        retVal = []
        coord = self.coord3d(position1D)
        for delta in [(-1, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
            i = delta[0] + coord[0]
            j = delta[1] + coord[1]
            k = delta[2] + coord[2]
            pos2D = self.coord1d(i, j, k)
            if self.coord_exists(i, j, k) and self.parentIndex[pos2D] < 0 and self.slowness[pos2D] != self.nullValue:
                retVal.append(pos2D)
        return retVal
    
    # Main propagation algorithm.
    # Expands from the seed cell, explores neighbors, and records parent links.
    def propagate(self):
        while self.narrowBand:  
            pos1D = heapq.heappop(self.narrowBand).index  # Pop cell with highest priority (lowest slowness)
            neighbors = self.neighbors(pos1D)  # Get valid neighbors
            if not neighbors:  # If no neighbors, this is an end cell
                self.endCells.append(pos1D)  
            for index in neighbors:  
                self.parentIndex[index] = pos1D  # Set parent
                cell = Cell(index, self.slowness[index])  
                heapq.heappush(self.narrowBand, cell) 

    # Compute the density of visits for each cell.
    # Density indicates how many paths pass through each cell.
    def compute_density(self):
        density = [0] * (self.nI * self.nJ * self.nK)
        for i in range(self.nI * self.nJ * self.nK):
            parent = self.parentIndex[i]
            while parent >= 0 and parent != self.nI * self.nJ * self.nK:
                density[parent] += 1
                parent = self.parentIndex[parent]
        
        # The following loop is just for debugging (can be uncommented if needed)
        # for i in range(self.nI):
        #     for j in range(self.nJ):
        #         for k in range(self.nK):
        #             index = i * self.nJ + j + k * self.nI * self.nJ
        #             print(f"Cell ({i}, {j}, {k}) density: {density[index]}")
        return density
    
    # Placeholder for reading slowness from file (not implemented)
    def readSlowness(self, file):
        pass

    # Return total number of cells in the grid
    def get_num_cells(self):
        return self.nI * self.nJ * self.nK

    # Return list of end cells (leaf nodes)
    def get_end_cells(self):
        return self.endCells.copy()
        
####################################################################################################################################################################
###################################################
# Class representing an edge between two points in the skeleton.
class Edge:
    def __init__(self, f, s):
        self.first = f  # Index of the first point
        self.second = s  # Index of the second point

###################################################
# Class for generating skeleton structure from the propagation result.
# It selects high-density points and connects them via edges.
class Skeleton:
    def __init__(self, propagation):
        self.propagation = propagation  # Propagation object containing parent links and grid info
        self.known_points = np.zeros(propagation.get_num_cells(), dtype=bool)  # Mark visited points
        self.nb_points = 0  # Number of skeleton points
        self.nb_edges = 0  # Number of edges
        self.edges = None  # Array of Edge objects
        self.coords = None  # Coordinates of skeleton points

    # Return number of skeleton points
    def getNbPoints(self):
        return self.nb_points

    # Return number of skeleton edges
    def getNbEdges(self):
        return self.nb_edges

    # Return coordinates of skeleton points
    def getCoords(self):
        return self.coords

    # Return list of edges
    def getEdges(self):
        return self.edges

    ###################################################
    # Main function to extract skeleton points and edges
    # based on a density threshold.
    # High-density cells are selected as skeleton points.
    # Parent-child relations are traced to construct edges.
    def follow_points(self, density_threshold):
        density = self.propagation.compute_density()  # Get density from propagation
        
        nijk = self.propagation.get_num_cells()  # Total number of grid cells
        
        # Mapping from full grid index to skeleton point index
        map_index = np.full(nijk, -1, dtype=int)
       
        # Count number of points exceeding density threshold
        for i in range(nijk):
            if density[i] >= density_threshold:
                self.nb_points += 1

        # Allocate arrays for skeleton points and edges
        self.edges = np.empty(self.nb_points - 1, dtype=object)
        self.coords = np.empty((self.nb_points, 3), dtype=int)

        index = 0
        # Store coordinates of skeleton points and fill mapping
        for i in range(nijk):
            if density[i] >= density_threshold: 
                coord = self.propagation.coord3d(i)
                self.coords[index] = coord
                index += 1
                map_index[i] = index
                
        end_cells = self.propagation.get_end_cells()  # Get end cells from propagation (not directly used here)
        
        k = 0  # Edge counter
        
        # Trace parents to form edges between skeleton points
        for i in range(nijk):
            start_point = i
            next_point = self.propagation.parentIndex[start_point]
            
            while next_point != nijk and next_point >= 0 and not self.known_points[start_point]:
                self.known_points[start_point] = True  # Mark point as visited
                if density[start_point] >= density_threshold:
                    # Add edge if both start and next point exceed threshold
                    self.edges[k] = Edge(map_index[start_point], map_index[next_point])
                    k += 1
                start_point = next_point
                next_point = self.propagation.parentIndex[start_point]
        
        self.nb_edges = k  # Store final number of edges
        print(k)  # Print number of edges (for debugging)

####################################################################################
# Class for reading property data, handling I/O, and writing skeletons or topology to file.
class TestSkeleton:
    def __init__(self):
        self.nullValue = 0.0
        self.nI = 0
        self.nJ = 0
        self.nK = 0  
        self.property = None

    ###################################################
    # Reads property file containing grid dimensions and slowness data.
    # File format: first line contains nI, nJ, nK, nullValue
    # Remaining lines contain one slowness value per cell.
    def readProperty(self, property_file):
        with open(property_file, 'r') as infile:
            line = infile.readline().strip()
            parts = line.split()
            self.nI = int(parts[0])
            print(self.nI)
            self.nJ = int(parts[1])
            self.nK = int(parts[2])  
            self.nullValue = float(parts[3])
            
            self.property = []
            for line in infile:
                self.property.append(float(line.strip()))
            
            # Validate number of read cells
            if len(self.property) != self.nI * self.nJ * self.nK:
                raise ValueError(
                    f"wrong file {property_file} read {len(self.property)} lines instead of {self.nI * self.nJ * self.nK}")
            
            # Optional debugging output
            if os.getenv("DEBUG"):
                print(f"read {property_file} ni {self.nI} nj {self.nJ} nk {self.nK} nullValue {self.nullValue}")

    ###################################################
    # Dispatcher for writing output depending on object type.
    def write(self, obj, withEdges=False):
        if isinstance(obj, Skeleton):
            self.writeSkeleton(obj, withEdges)
        elif isinstance(obj, Topology):
            self.writeTopology(obj)

    ###################################################
    # Writes skeleton points (and optionally edges) to file.
    # Output format: coordinates list, followed by SEGMENT section.
    def writeSkeleton(self, skeleton, withEdges):
        index = 1
        with open('skeleton.skel', 'w') as f:
            for p in range(skeleton.getNbPoints()):
                f.write(f"{index}, {skeleton.getCoords()[p][0]}, {skeleton.getCoords()[p][1]}, {skeleton.getCoords()[p][2]}\n")
                index += 1
            
            if withEdges:
                f.write("SEGMENT\n")
                # The actual edge writing is currently commented out
                # for e in range(skeleton.getNbEdges()):
                #     edge = skeleton.getEdges()[e]
                #     f.write(f"{edge.first}, {edge.second}\n")
            
            f.write("END OF SKELETON\n")

    ###################################################
    # Writes topology branch information to file.
    # This method appends data after skeleton point section.
    def writeTopology(self, topology):
        with open('skeleton.skel', 'a') as f:
            f.write("SEGMENT\n")
            done = set()
            count = 0
            for i in range(topology.nb_branches):
                b = topology.pt_to_branch[i]
                if b is None or b in done:
                    continue
                count += 1
                done.add(b)
                r = topology.rank(b)
                s = b.first_point()
                for j in range(1, len(b.pts)):
                    l = b.pts[j]
                    f.write(f"{s}, {l}, {r}, {count}\n")
                    s = l

    # Get total number of grid cells
    def getNumCells(self):
        return self.nI * self.nJ * self.nK

    # Get grid size along I
    def getNI(self):
        return self.nI

    # Get grid size along J
    def getNJ(self):
        return self.nJ

    # Get grid size along K
    def getNK(self):
        return self.nK  

    # Get slowness property array
    def getProperty(self):
        return self.property

    # Get null value representing invalid data
    def getNullValue(self):
        return self.nullValue


###########################################################################################################
# Class representing an edge between two points in the skeleton.
class Edge:
    def __init__(self, first, second):
        self.first = first  # Index of the first point
        self.second = second  # Index of the second point

###################################################
# Class representing a branch composed of a sequence of points.
# A branch can be created from a single edge or a list of points.
class Branch:
    def __init__(self, edge=None, edges=None):
        self.pts = []  # List of point indices representing the branch path
        if edge:  # Create branch from a single edge
            self.pts.append(edge.second)
            self.pts.append(edge.first)
        elif edges:  # Create branch from a list of points
            self.pts = edges
            if len(edges) < 2:
                print("Error - branch without any edges")

    # Create a copy of the branch
    def clone(self):
        return Branch(edges=self.pts.copy())

    # Check if branch contains the given point (excluding first point)
    def contains(self, point):
        return point in self.pts[1:]

    # Return the first point of the branch
    def first_point(self):
        return self.pts[0]

    # Return the last point of the branch
    def last_point(self):
        return self.pts[-1]

    # Create a new branch starting from given point to the end
    def from_point_to_end(self, point):
        last_points = []
        find_pt = False
        for pt in self.pts:
            if pt == point:
                find_pt = True
            if find_pt:
                last_points.append(pt)
        if len(last_points) > 1:
            return Branch(edges=last_points)
        return None

    # Create a new branch from start to the given point
    def from_start_to_point(self, point):
        first_points = []
        for pt in self.pts:
            first_points.append(pt)
            if pt == point:
                break
        if len(first_points) > 1:
            return Branch(edges=first_points)
        else:
            return None

    # Append another branch to current branch if they connect correctly
    def append(self, ends):
        if self.last_point() != ends.first_point():
            print("error - mismatch in Branch::append")
            return
        self.pts.extend(ends.pts[1:])

###################################################
# Class responsible for generating topology of skeleton.
# It merges edges into branches, handles connectivity and hierarchy.
class Topology:
    def __init__(self, edges, nb_edges, pt_max):
        self.nb_branches = pt_max + 1  # Total possible branches (one for each point index)
        self.edges = edges  # Input list of edges
        self.nb_edges = nb_edges  # Number of edges
        self.pt_max = pt_max  # Maximum point index

        self.pt_to_branch = [None] * self.nb_branches  # Mapping from point index to branch object
        print(self.pt_to_branch)

        # Iterate over all edges to build branches
        for edge in edges:
            b1 = Branch(edge=edge)
            last_point = b1.last_point()
            print(f"Last point of b1: {last_point}")

            b2 = self.pt_to_branch[b1.last_point()]  # Check if this point already has an existing branch

            if b2:
                b1.append(b2)
                del b2

            father = self.pt_to_branch[b1.first_point()]  # Find possible parent branch

            self.update_pt_to_branch(b1)  # Update mapping with new branch
            self.merge(father, b1)  # Attempt to merge branches

    # Destructor to cleanup references
    def __del__(self):
        for b in self.pt_to_branch:
            if b:
                for pt in b.pts:
                    self.pt_to_branch[pt] = None
                del b

    ###################################################
    # Attempt to merge two branches together
    def merge(self, b1, b2):
        if b1 == b2 or not b1 or not b2:
            return
        if len(b1.pts) == 1:
            print("error merge")
            return

        pt = b2.first_point()
        if pt == b1.last_point():
            b1.append(b2)
            self.update_pt_to_branch(b1)
            father = self.pt_to_branch[b1.first_point()]
            del b2
            if father:
                self.merge(father, b1)
            return

        b3 = b1.from_start_to_point(pt)
        if b3 is not None:
            print("b3", b3.pts)
            b3.append(b2)
            if len(b1.pts) >= len(b3.pts):
                del b3
                return

            self.update_pt_to_branch(b3)

        b4 = b1.from_point_to_end(pt)
        if b4 is not None:
            if len(b4.pts) == 1:
                print("error merge")
                return

            self.update_pt_to_branch(b4)
            del b1
            del b2
            father = self.pt_to_branch[b3.first_point()] if b3 else None
            if father:
                self.merge(father, b3)

    ###################################################
    # Update the point-to-branch mapping for all points in a branch
    def update_pt_to_branch(self, b):
        if b is None:
            print("Error: Attempted to update with a None branch.")
            return
        if not self.pt_to_branch[b.first_point()]:
            self.pt_to_branch[b.first_point()] = b
        for pt in b.pts[1:]:
            self.pt_to_branch[pt] = b

    ###################################################
    # Compute the hierarchical rank (depth) of a branch
    def rank(self, b):
        r = 0
        while b:
            r += 1
            b2 = self.pt_to_branch[b.first_point()]
            if b == b2:
                break
            b = b2
        return r

######################################################################################

# Entry point of the program that handles full pipeline execution
if __name__ == "__main__":
    
    # Set parameters
    grid_file = "liantongti/skeletonization_technique-main/LIANTONGTI/LIANTONGTI/新建文件夹/DATA/imput/DATA.txt"  # Path to input property file
    threshold = 1000         # Density threshold for skeleton extraction
    seed_i, seed_j, seed_k = 0, 0, 0  # Seed point coordinates

    # Create object to read property data
    reader = TestSkeleton()
    reader.readProperty(grid_file)

    # Initialize propagation using property data
    prop = Propagation(reader.getProperty(),
                       reader.getNullValue(),
                       reader.getNI(),
                       reader.getNJ(),
                       reader.getNK(),
                       seed_i, seed_j, seed_k)
    prop.propagate()

    # Generate skeleton from propagation result
    skeleton = Skeleton(prop)
    skeleton.follow_points(threshold)

    # Write skeleton file
    reader.write(skeleton, False)

    # Generate topology from skeleton edges
    topology = Topology(skeleton.getEdges(), skeleton.getNbEdges(), skeleton.getNbPoints())
    reader.write(topology)

###################################################
# Read skeleton nodes and segments from file
def read_skeleton_file(filename):
    with open(filename, 'r') as file:
        data = file.read()

    # Split content into nodes and segments
    skeleton_part, segment_part = data.split('END OF SKELETON')

    nodes = []
    for line in skeleton_part.strip().splitlines():
        line = line.replace(',', '')
        values = line.split()
        if len(values) == 4:
            index, x, y, z = map(float, values)
            nodes.append((int(index), x, y, z))

    segments = []
    for line in segment_part.strip().splitlines():
        line = line.replace(',', '')
        values = line.split()
        if len(values) == 4:
            start_idx, end_idx, rank, branch_idx = map(int, values)
            segments.append((start_idx, end_idx, rank, branch_idx))

    return nodes, segments

###################################################
# Visualize 3D skeleton with branch coloring and labels
def visualize_skeleton(nodes, segments):
    # Extract node coordinates
    node_indices, x_coords, y_coords, z_coords = zip(*nodes)
    x_coords, y_coords, z_coords = np.array(x_coords), np.array(y_coords), np.array(z_coords)

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot all skeleton nodes
    ax.scatter(x_coords, y_coords, z_coords, c='lightgrey', marker='.', s=2)

    # Define distinct colors for branches
    branch_colors = [
        'red', 'purple', 'green', 'blue', 'orange', 'cyan', 'magenta', 
        'yellow', 'brown', 'pink', 'olive', 'teal', 'navy', 'gold', 
        'darkgreen', 'darkred', 'chocolate', 'crimson', 'lime', 'indigo'
    ]

    # Map branch indices to colors
    branch_indices = sorted(set(seg[3] for seg in segments))
    branch_idx_to_color = {
        branch_idx: branch_colors[i] for i, branch_idx in enumerate(branch_indices) if i < len(branch_colors)
    }

    # Prevent duplicate segment drawing while keeping minimal rank
    drawn_segments = {}
    for seg in segments:
        start_idx, end_idx, rank, branch_idx = seg
        seg_key = tuple(sorted([start_idx, end_idx]))

        if seg_key in drawn_segments:
            existing_rank = drawn_segments[seg_key]['rank']
            if rank < existing_rank:
                drawn_segments[seg_key] = {'rank': rank, 'branch_idx': branch_idx}
        else:
            drawn_segments[seg_key] = {'rank': rank, 'branch_idx': branch_idx}

    # Set equal aspect ratio for 3D visualization
    x_range, y_range, z_range = [x_coords.min(), x_coords.max()], [y_coords.min(), y_coords.max()], [z_coords.min(), z_coords.max()]
    max_range = max(x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0])
    center_x, center_y, center_z = (x_range[1] + x_range[0]) / 2, (y_range[1] + y_range[0]) / 2, (z_range[1] + z_range[0]) / 2
    ax.set_xlim(center_x - max_range / 2, center_x + max_range / 2)
    ax.set_ylim(center_y - max_range / 2, center_y + max_range / 2)
    ax.set_zlim(center_z - max_range / 2, center_z + max_range / 2)

    # Draw all segments with branch labels
    segment_display_count = 0
    for seg_key, seg_info in drawn_segments.items():
        start_idx, end_idx = seg_key
        rank, branch_idx = seg_info['rank'], seg_info['branch_idx']
        color = branch_idx_to_color.get(branch_idx, 'lightgrey')

        start_node = nodes[start_idx - 1]
        end_node = nodes[end_idx - 1]

        mid_x, mid_y, mid_z = (start_node[1] + end_node[1]) / 2, (start_node[2] + end_node[2]) / 2, (start_node[3] + end_node[3]) / 2

        # Slight random offset to avoid label overlap
        offset = 0.5
        display_x = mid_x + offset * (np.random.rand() - 0.5)
        display_y = mid_y + offset * (np.random.rand() - 0.5)
        display_z = mid_z + offset * (np.random.rand() - 0.5)

        if segment_display_count % 12 == 0:
            ax.text(display_x, display_y, display_z, str(branch_idx), color=color, fontsize=12, ha='center', va='center')

        ax.plot([start_node[1], end_node[1]], 
                [start_node[2], end_node[2]], 
                [start_node[3], end_node[3]], 
                c=color, linestyle='--', linewidth=2)

        segment_display_count += 1

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    plt.title('3D Skeleton Visualization with Branch Levels')
    plt.show()



###################################################
# Entry point to load skeleton file and visualize 3D skeleton
if __name__ == "__main__":
    # Specify the skeleton file path
    skeleton_file = 'skeleton.skel'  # Input file generated from skeletonization process

    # Load skeleton data: nodes (points) and segments (edges)
    nodes, segments = read_skeleton_file(skeleton_file)

    # Visualize the 3D skeleton structure with branch indices
    visualize_skeleton(nodes, segments)




