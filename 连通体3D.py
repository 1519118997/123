import os
import sys
import heapq
import numpy as np
import heapq
##########################################################################################################################################
class Cell:
    def __init__(self, index=0, value=0.0):
        self.index = index  # 单元格的索引
        self.value = -value  # 单元格的值，取负

    # 定义小于运算符，用于堆排序
    def __lt__(self, other):
        return self.value < other.value


class Propagation:
    def __init__(self, prop, nullV, ni, nj, nk, seed_i, seed_j, seed_k):
        self.nullValue = nullV
        self.seed_i = seed_i
        self.seed_j = seed_j
        self.seed_k = seed_k
        self.nI = ni
        self.nJ = nj
        self.nK = nk
        self.parentIndex = [-1] * (ni * nj * nk)
        self.slowness = prop
        self.narrowBand = []
        self.endCells = []

        cell = Cell()
        cell.index = self.coord1d(seed_i, seed_j, seed_k)
        self.parentIndex[cell.index] = ni * nj * nk
        heapq.heappush(self.narrowBand, Cell(cell.index, self.slowness[cell.index]))

    def coord3d(self, index):
        out = [0, 0, 0]
        out[0] = index // (self.nJ * self.nK)
        out[1] = (index % (self.nJ * self.nK)) // self.nK
        out[2] = index % self.nK
        return out
    

    def coord1d(self, i, j, k):
        return i * self.nJ * self.nK + j * self.nK + k

    def coord_exists(self, i, j, k):
        return 0 <= i < self.nI and 0 <= j < self.nJ and 0 <= k < self.nK

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
    
        # 进行传播过程
    def propagate(self):
        while self.narrowBand:  # 当堆不为空时
            pos1D = heapq.heappop(self.narrowBand).index  # 弹出堆顶元素的索引
            neighbors = self.neighbors(pos1D)  # 获取邻居
            if not neighbors:  # 如果没有邻居
                self.endCells.append(pos1D)  # 将当前元素添加到终止单元格列表中
            for index in neighbors:  # 遍历所有邻居
                self.parentIndex[index] = pos1D  # 设置邻居的父索引
                cell = Cell(index, self.slowness[index])  # 创建新的 Cell 对象
                heapq.heappush(self.narrowBand, cell)  # 将新的 Cell 压入堆中

    def compute_density(self):
        density = [0] * (self.nI * self.nJ * self.nK)
        for i in range(self.nI * self.nJ * self.nK):
            parent = self.parentIndex[i]
            while parent >= 0 and parent != self.nI * self.nJ * self.nK:
                density[parent] += 1

                parent = self.parentIndex[parent]
                
        for i in range(self.nI):
            for j in range(self.nJ):
                for k in range(self.nJ):
                    index = i * self.nJ + j + k * self.nI * self.nJ
                    # print(f"Cell ({i}, {j}, {k}) density: {density[index]}")
        return density
    
    
    def readSlowness(self, file):
        pass

    #
    def get_num_cells(self):
        return self.nI * self.nJ * self.nK

    # 
    def get_end_cells(self):
        return self.endCells.copy()
        

####################################################################################################################################################################
class Edge:
    def __init__(self, f, s):
        self.first = f  # 起始点索引
        self.second = s  # 结束点索引

class Skeleton:
    def __init__(self, propagation):
        self.propagation = propagation  # 传递算法的实例
        self.known_points = np.zeros(propagation.get_num_cells(), dtype=bool)  # 记录已知的点，初始为全False
        self.nb_points = 0  # 骨架中的点数量
        self.nb_edges = 0  # 骨架中的边数量
        self.edges = None  # 用于存储边的数组
        self.coords = None  # 用于存储坐标的数组

    def getNbPoints(self):
        return self.nb_points  # 返回点数量

    def getNbEdges(self):
        return self.nb_edges  # 返回边数量

    def getCoords(self):
        return self.coords  # 返回点的坐标

    def getEdges(self):
        return self.edges  # 返回边

    def follow_points(self, density_threshold):
        density = self.propagation.compute_density()  # 计算密度
        # print(density)
        nijk = self.propagation.get_num_cells()  # 获取单元格总数
        # print(nij)

        # 初始化索引映射，初始值为-1
        map_index = np.full(nijk, -1, dtype=int)
        # print(map_index)
        
        # 计算符合密度阈值的点数量
        for i in range(nijk):
            if density[i] >= density_threshold:
                self.nb_points += 1

        # 初始化边和坐标数组
        self.edges = np.empty(self.nb_points - 1, dtype=object)
      
        self.coords = np.empty((self.nb_points, 3), dtype=int)

        index = 0
        # 将符合密度阈值的点的坐标存储到 coords 数组中，并更新索引映射
        for i in range(nijk):
            if density[i] >= density_threshold: 
                coord = self.propagation.coord3d(i)
                self.coords[index] = coord
                index += 1
                map_index[i] = index
                # print(map_index)

        end_cells = self.propagation.get_end_cells()  # 获取结束单元格
        k = 0
        # 遍历所有单元格，跟随传播路径创建边
        for i in range(nijk):
            start_point = i
            next_point = self.propagation.parentIndex[start_point]
            # print(next_point)
            # 遍历路径直到找到已知点或超出边界
            while next_point != nijk and next_point >= 0 and not self.known_points[start_point]:
                self.known_points[start_point] = True  # 标记该点为已知
                if density[start_point] >= density_threshold:
                    # 创建边并存储
                    self.edges[k] = Edge(map_index[start_point], map_index[next_point])
                    k += 1
                start_point = next_point
                next_point = self.propagation.parentIndex[start_point]
        self.nb_edges = k  # 更新边数量
        print(k)



###################################################################################################################################################################
class TestSkeleton:
    def __init__(self):
        self.nullValue = 0.0
        self.nI = 0
        self.nJ = 0
        self.nK = 0  # 添加第三维
        self.property = None

    def readProperty(self, property_file):
        with open(property_file, 'r') as infile:
            # 读取文件头
            line = infile.readline().strip()
            parts = line.split()
            self.nI = int(parts[0])
            print(self.nI)
            self.nJ = int(parts[1])
            self.nK = int(parts[2])  # 读取第三维
            self.nullValue = float(parts[3])
            # 分配属性数组的内存
            self.property = []
            # 读取属性值
            for line in infile:
                self.property.append(float(line.strip()))
            # 检查文件中读取的值的数量是否正确
            if len(self.property) != self.nI * self.nJ * self.nK:
                raise ValueError(
                    f"wrong file {property_file} read {len(self.property)} lines instead of {self.nI * self.nJ * self.nK}")
            # 仅打印前10个属性值
            # 如果设置了 DEBUG 环境变量，输出调试信息
            if os.getenv("DEBUG"):
                print(f"read {property_file} ni {self.nI} nj {self.nJ} nk {self.nK} nullValue {self.nullValue}")

    def write(self, obj, withEdges=False):
        if isinstance(obj, Skeleton):
            self.writeSkeleton(obj, withEdges)
        elif isinstance(obj, Topology):
            self.writeTopology(obj)

    def writeSkeleton(self, skeleton, withEdges):
        index = 1
        # 在写入骨架数据的同时打开文件
        with open('skeleton.skel', 'w') as f:
            # 写入骨架的坐标
            for p in range(skeleton.getNbPoints()):
                f.write(f"{index}, {skeleton.getCoords()[p][0]}, {skeleton.getCoords()[p][1]}, {skeleton.getCoords()[p][2]}\n")
                index += 1
            
            # 如果需要写入边的信息
            if withEdges:
                f.write("SEGMENT\n")
                # for e in range(skeleton.getNbEdges()):
                #     edge = skeleton.getEdges()[e]
                #     f.write(f"{edge.first}, {edge.second}\n")
            
            # 添加分隔词
            f.write("END OF SKELETON\n")

    def writeTopology(self, topology):
        # 在写入拓扑数据的同时打开文件
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
                # 输出分支中的所有点
                for j in range(1, len(b.pts)):
                    l = b.pts[j]
                    f.write(f"{s}, {l}, {r}, {count}\n")
                    s = l


    def getNumCells(self):
        return self.nI * self.nJ * self.nK

    def getNI(self):
        return self.nI

    def getNJ(self):
        return self.nJ

    def getNK(self):
        return self.nK  # 添加获取第三维的方法

    def getProperty(self):
        return self.property

    def getNullValue(self):
        return self.nullValue


###########################################################################################################
class Edge:
    def __init__(self, first, second):
        self.first = first  # first 是一个三维坐标 (x, y, z)
        self.second = second  # second 是一个三维坐标 (x, y, z)


class Branch:
    def __init__(self, edge=None, edges=None):
        self.pts = []  # 存储三维点的列表
        if edge:  # 如果提供了一条边
            self.pts.append(edge.second)  # 添加边的第二个端点
            self.pts.append(edge.first)  # 添加边的第一个端点
        elif edges:  # 如果提供了一组点
            self.pts = edges  # 直接使用这些点
            if len(edges) < 2:
                print("Error - branch without any edges")
    
    def clone(self):
        return Branch(edges=self.pts.copy())  # 返回当前分支的副本

    def contains(self, point):
        return point in self.pts[1:]  # 检查点是否在列表中（不包括第一个点）

    def first_point(self):
        return self.pts[0]  # 返回第一个点

    def last_point(self):
        return self.pts[-1]  # 返回最后一个点
    
    def from_point_to_end(self, point):
        last_points = []  # 从指定点到末尾的点列表
        find_pt = False  # 标记点是否已找到
        for pt in self.pts:  # 遍历点
            if pt == point:
                find_pt = True
            if find_pt:
                last_points.append(pt)
        if len(last_points) > 1:
            return Branch(edges=last_points)  # 返回新的分支
        else:
            return None
        
    def from_start_to_point(self, point):
        first_points = []  # 从起点到指定点的点列表
        for pt in self.pts:  # 遍历点
            first_points.append(pt)
            if pt == point:
                break
        if len(first_points) > 1:
            return Branch(edges=first_points)  # 返回新的分支
        else:
            return None
        
    def append(self, ends):
        if self.last_point() != ends.first_point():  # 检查最后一个点是否匹配
            print("error - mismatch in Branch::append")
            return
        self.pts.extend(ends.pts[1:])  # 合并点列表（不包括第一个点）


class Topology:
    def __init__(self, edges, nb_edges, pt_max):
        self.nb_branches = pt_max + 1  # 初始化分支数量
        self.edges = edges  # 初始化边列表
        self.nb_edges = nb_edges  # 初始化边数量
        self.pt_max = pt_max  # 初始化最大点索引

        self.pt_to_branch = [None] * self.nb_branches  # 初始化点到分支的映射
        print(self.pt_to_branch)

        for edge in edges:
            b1 = Branch(edge=edge)  # 创建新的分支
            
            last_point = b1.last_point()
            print(f"Last point of b1: {last_point}")

            b2 = self.pt_to_branch[b1.last_point()]  # 获取当前分支最后一个点对应的分支

            if b2:
                b1.append(b2)  # 合并分支
                del b2

            father = self.pt_to_branch[b1.first_point()]  # 获取当前分支第一个点对应的分支

            self.update_pt_to_branch(b1)  # 更新点到分支的映射
            self.merge(father, b1)  # 合并分支
            
    def __del__(self):
        for b in self.pt_to_branch:  # 遍历所有分支
            if b:
                for pt in b.pts:  # 遍历分支中的所有点
                    self.pt_to_branch[pt] = None  # 将点对应的分支设为 None
                del b

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


    def update_pt_to_branch(self, b):
        if b is None:
            print("Error: Attempted to update with a None branch.")
            return
        if not self.pt_to_branch[b.first_point()]:
            self.pt_to_branch[b.first_point()] = b
        for pt in b.pts[1:]:
            self.pt_to_branch[pt] = b

    def rank(self, b):
        r = 0
        while b:
            r += 1
            b2 = self.pt_to_branch[b.first_point()]
            if b == b2:
                break
            b = b2
        return r


##################################################################################################

#####################################################################################################################################################################

if __name__ == "__main__":
    # nullV = -9999
    # seed_i = 0
    # seed_j = 0
    # seed_k = 0
    # ni = 2
    # nj = 2
    # nk = 2
    
    # prop = [15.5, 12, 10, 9.5, 14.5, 11, 8, 7.5]
    # propagation = Propagation(prop, nullV, ni, nj, nk, seed_i, seed_j, seed_k)
    # propagation.propagate()
    # propagation.compute_density()
    
    # skeleton = Skeleton(propagation)
  
    # density_threshold = 1
    # skeleton.follow_points(density_threshold)
    # reader = TestSkeleton()
    # reader.write(skeleton, False)
    
    # topology = Topology(skeleton.getEdges(), skeleton.getNbEdges(), skeleton.getNbPoints())
    # reader.write(topology)
    grid_file="C://Users/002022/Desktop/s.txt"
    threshold = 1000
    seed_i = 0
    seed_j = 0
    seed_k = 0
    reader = TestSkeleton()
    reader.readProperty(grid_file)
    prop=Propagation(reader.getProperty(),reader.getNullValue(),reader.getNI(),reader.getNJ(),reader.getNK(),seed_i,seed_j,seed_k)
    prop.propagate()
    skeleton=Skeleton(prop)
    skeleton.follow_points(threshold)
    reader.write(skeleton, False)
    topology = Topology(skeleton.getEdges(), skeleton.getNbEdges(), skeleton.getNbPoints())
    reader.write(topology)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.colors as mcolors

# 读取骨架文件的函数
def read_skeleton_file(filename):
    with open(filename, 'r') as file:
        data = file.read()

    # 将数据分割为 "END OF SKELETON" 和 "SEGMENT" 两部分
    skeleton_part, segment_part = data.split('END OF SKELETON')

    # 处理节点数据 (在 "END OF SKELETON" 之前的部分)
    nodes = []
    for line in skeleton_part.strip().splitlines():
        # 假设节点格式为：index x y z
        line = line.replace(',', '')  # 去除逗号
        values = line.split()
        if len(values) == 4:
            index, x, y, z = map(float, values)
            nodes.append((int(index), x, y, z))

    # 处理段数据 (在 "SEGMENT" 之后的部分)
    segments = []
    for line in segment_part.strip().splitlines():
        # 假设段格式为：start_node, end_node, rank, branch_idx
        line = line.replace(',', '')  # 去除逗号
        values = line.split()
        if len(values) == 4:
            start_idx, end_idx, rank, branch_idx = map(int, values)
            segments.append((start_idx, end_idx, rank, branch_idx))

    return nodes, segments

# 可视化骨架的函数
def visualize_skeleton(nodes, segments):
    # 提取节点坐标
    node_indices, x_coords, y_coords, z_coords = zip(*nodes)
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    z_coords = np.array(z_coords)

    # 创建图形和3D轴
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制节点，浅灰色
    ax.scatter(x_coords, y_coords, z_coords, c='lightgrey', marker='.', s=2)

    # 定义分支颜色，按照顺序分配
    branch_colors = ['red', 'purple', 'green', 'blue', 'orange']

    # 找到所有存在的 branch_idx
    branch_indices = sorted(set(seg[3] for seg in segments))

    # 创建 branch_idx 到颜色的映射
    branch_idx_to_color = {}
    for i, branch_idx in enumerate(branch_indices):
        if i < len(branch_colors):
            branch_idx_to_color[branch_idx] = branch_colors[i]

    # 用于记录绘制过的线段及其最高优先级（数字越小等级越高）
    drawn_segments = {}

    # 第一步：遍历所有段，找到每条线段的最低等级（最高优先级）和分支编号
    for seg in segments:
        start_idx, end_idx, rank, branch_idx = seg
        seg_key = tuple(sorted([start_idx, end_idx]))  # 确保起点和终点的顺序一致

        # 更新已绘制线段，保留优先级更高的分支
        if seg_key in drawn_segments:
            existing_rank = drawn_segments[seg_key]['rank']
            if rank < existing_rank:
                drawn_segments[seg_key] = {'rank': rank, 'branch_idx': branch_idx}
        else:
            drawn_segments[seg_key] = {'rank': rank, 'branch_idx': branch_idx}

    # 设置轴的范围，保持等比缩放
    x_range = [x_coords.min(), x_coords.max()]
    y_range = [y_coords.min(), y_coords.max()]
    z_range = [z_coords.min(), z_coords.max()]

    # 计算最大范围和中心
    max_range = max(x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0])
    center_x = (x_range[1] + x_range[0]) / 2
    center_y = (y_range[1] + y_range[0]) / 2
    center_z = (z_range[1] + z_range[0]) / 2

    # 设置坐标轴的范围
    ax.set_xlim(center_x - max_range / 2, center_x + max_range / 2)
    ax.set_ylim(center_y - max_range / 2, center_y + max_range / 2)
    ax.set_zlim(center_z - max_range / 2, center_z + max_range / 2)

    # 第二步：根据分支的等级分配颜色，并显示分支等级数字
    segment_display_count = 0  # 用于稀疏显示分支等级
    for seg_key, seg_info in drawn_segments.items():
        start_idx, end_idx = seg_key
        rank = seg_info['rank']
        branch_idx = seg_info['branch_idx']

        # 根据实际的 branch_idx 分配颜色
        color = branch_idx_to_color.get(branch_idx, 'lightgrey')  # 未映射的分支使用灰色

        # 计算显示分支数字的位置（稍微偏离骨架）
        start_node = nodes[start_idx - 1]
        end_node = nodes[end_idx - 1]

        # 计算中点坐标
        mid_x = (start_node[1] + end_node[1]) / 2
        mid_y = (start_node[2] + end_node[2]) / 2
        mid_z = (start_node[3] + end_node[3]) / 2

        # 计算偏移量
        offset = 0.5
        display_x = mid_x + offset * (np.random.rand() - 0.5)  # 随机生成偏移量
        display_y = mid_y + offset * (np.random.rand() - 0.5)
        display_z = mid_z + offset * (np.random.rand() - 0.5)

        # 每隔12个分支显示一次分支等级数字
        if segment_display_count % 12 == 0:
            # 在中点位置显示分支等级数字
            ax.text(display_x, display_y, display_z, str(branch_idx), color=color, fontsize=12, ha='center', va='center')

        # 绘制虚线段
        ax.plot([start_node[1], end_node[1]], 
                [start_node[2], end_node[2]], 
                [start_node[3], end_node[3]], 
                c=color, linestyle='--', linewidth=2)

        segment_display_count += 1  # 更新显示计数

    # 设置轴标签
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')

    # 显示图形
    plt.title('3D Skeleton Visualization with Branch Levels')
    plt.show()

# 主程序
if __name__ == "__main__":
    skeleton_file = 'skeleton.skel'  # 替换为你的实际文件路径
    nodes, segments = read_skeleton_file(skeleton_file)
    visualize_skeleton(nodes, segments)



