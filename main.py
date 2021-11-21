import matplotlib
import numpy as np
import shapefile
from sklearn.preprocessing import normalize
import math
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


class Shape:
    def __init__(self, folder, name):
        self.folder = folder
        self.name = name
        self.shp = self.get_shp()
        self.points = self.shp.points

    def get_reader(self):
        pole_shx = open(f"{self.folder}/{self.name}.shx", "rb")
        pole_shp = open(f"{self.folder}/{self.name}.shp", "rb")
        pole_dbf = open(f"{self.folder}/{self.name}.dbf", "rb")
        return shapefile.Reader(shx=pole_shx, shp=pole_shp, dbf=pole_dbf)

    def get_shp(self):
        with self.get_reader() as polygon:
            shp = polygon.shape(0)
            print(polygon,
                  f"type num={polygon.shapeType}\n",
                  f"len={len(polygon)}\n",
                  f"bounding box area={polygon.bbox}\n"
                  f"shapes={len(polygon.shapes())}\n"
                  f"shape box={['%.3f' % coord for coord in shp.bbox]}\n"
                  f"shape parts={[coord for coord in shp.parts]}\n"
                  f"shape points num={len(shp.points)}\n"
                  f"shape points={[coord for coord in shp.points]}\n"
                  )
            return shp

    def split_into_2dlist(self):
        list_x, list_y = zip(*self.points)
        return list_x, list_y


class Plotter:

    def __init__(self, boundary_shape: Shape):
        self.boundary_shape = boundary_shape

    def open_shp(self, shape):
        list_x, list_y = shape.split_into_2dlist()
        plt.scatter(list_x, list_y)
        plt.plot(list_x, list_y)

    def draw_solution(self, solution_shape):
        self.open_shp(solution_shape)
        self.open_shp(self.boundary_shape)
        plt.show()


def demo_solution():
    root = "Trimble/AgGPS/Data/Hackathon/Hackathon/3_2"
    boundary = Shape(root, "Boundary")
    solution = Shape(root, "LineFeature")
    plotter = Plotter(boundary)
    plotter.draw_solution(solution)


def get_norm(vector):
    return normalize(vector[:, np.newaxis], axis=0).ravel()


def has_turn(vector1, vector2):
    nv1 = get_norm(vector1)
    nv2 = get_norm(vector2)
    x_same_sign = (nv1[0] >= 0) == (nv2[0] >= 0)
    y_same_sign = (nv1[1] >= 0) == (nv2[1] >= 0)
    return not (x_same_sign and y_same_sign)


def split_into_segments(boundary_shape, set_plot=False):
    boundary_list = []
    start_vector = None
    sp = None
    for i, point in enumerate(boundary_shape.points):
        if i == 0:
            prev_point = point
            continue
        if i == 1:
            sp = point
        curr_vector = np.array(point) - np.array(prev_point)
        if start_vector is None:
            start_vector = curr_vector
            prev_point = point
            continue
        if has_turn(start_vector, curr_vector):
            boundary_list.append([point])
            boundary_list[-1].append(prev_point)
            start_vector = curr_vector
        else:
            boundary_list[-1].append(point)
        prev_point = point
        if i == (len(boundary_shape.points) - 1):
            boundary_list[-1].append(sp)
    if set_plot:
        for boundary in boundary_list:
            list_x, list_y = zip(*boundary)
            plt.scatter(list_x, list_y, alpha=0.3)
            plt.plot(list_x, list_y, alpha=0.3)
    return boundary_list


def find_most_long_vector(shape, set_plot=False):
    max_norm = None
    max_vect = None
    max_segm = None
    for segm in split_into_segments(shape, set_plot):
        start_point = segm[0]
        end_point = segm[-1]
        vect = np.array(end_point) - np.array(start_point)
        curr_norm = np.linalg.norm(vect)
        if not max_norm or curr_norm > max_norm:
            max_norm = curr_norm
            max_vect = vect
            max_segm = segm
    if set_plot:
        list_x, list_y = zip(*max_segm)
        plt.scatter(list_x, list_y, alpha=0.5)
        plt.plot(list_x, list_y, alpha=0.5)
    return max_vect, max_segm


def get_box_middles(shape, set_plot=False):
    bbox = shape.shp.bbox
    # like (0, 0):
    a = np.array([bbox[0], bbox[1]])
    # like (1, 1):
    b = np.array([bbox[2], bbox[3]])
    l = b - a
    lx = l[0]
    ly = l[1]
    if set_plot:
        rect = matplotlib.patches.Rectangle(a, width=lx, height=ly, color="purple", linewidth=2, fill=False)
        plt.gca().add_patch(rect)
    horizontal_middle = a[0] + (lx / 2)
    vertical_middle = a[1] + (ly / 2)
    return horizontal_middle, vertical_middle

def chose_clone_line_direction(shape, most_long_vect, max_segm):
    is_horisontal_direction = np.linalg.norm(most_long_vect)[0] <= math.cos(45)
    horizontal_middle, vertical_middle = get_box_middles(shape)
    x_list, y_list = zip(*max_segm)
    if is_horisontal_direction:
        y_list



pole = Shape("Trimble", "Pole")
most_long_vect, max_segm = find_most_long_vector(pole, True)
chose_clone_line_direction(pole, most_long_vect, max_segm)

plt.show()
