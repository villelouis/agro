import matplotlib
import numpy as np
import shapefile
from sklearn.preprocessing import normalize

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


def find_bound(boundary_shape):
    boundary_list = []
    start_vector = None
    curr_vector = None
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
    for boundary in boundary_list:
        list_x, list_y = zip(*boundary)
        plt.scatter(list_x, list_y)
        plt.plot(list_x, list_y)
    plt.show()


find_bound(Shape("Trimble", "Pole"))
