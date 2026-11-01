# coding: utf-8
import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


def viz_lmk(img_, vps, **kwargs):
    """可视化点"""
    lineType = kwargs.get("lineType", cv2.LINE_8)  # cv2.LINE_AA
    img_for_viz = img_.copy()
    for pt in vps:
        cv2.circle(
            img_for_viz,
            (int(pt[0]), int(pt[1])),
            radius=kwargs.get("radius", 1),
            color=(0, 255, 0),
            thickness=kwargs.get("thickness", 1),
            lineType=lineType,
        )
    return img_for_viz


def plot_3d_scatter(data, xlabel='X Label', ylabel='Y Label', zlabel='Z Label', filename=None):
    """
    绘制3D散点图。

    参数:
        data (numpy.ndarray): 形状为 (n, 3) 的数组，其中 n 是样本数量。
        xlabel (str): X 轴的标签文本，默认是 'X Label'。
        ylabel (str): Y 轴的标签文本，默认是 'Y Label'。
        zlabel (str): Z 轴的标签文本，默认是 'Z Label'。
        filename (str): 保存图像的文件名，默认为 None（不保存）。
    """
    # 创建一个图形实例
    fig = plt.figure()

    # 创建一个三维的绘图工具
    ax = fig.add_subplot(111, projection='3d')

    # 假设我们根据第三个特征来决定颜色
    colors = data[:, 2]

    # 使用对比度高的颜色映射
    cmap = 'Set1'

    # 绘制三维散点图
    sc = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors, cmap=cmap)

    # 保存图像到文件
    if filename:
        fig.savefig(filename, bbox_inches='tight')

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

def plot_vectors(points, vector_diffs, filename=None):
    assert points.shape[0] == vector_diffs.shape[0], "Points and vector diffs must have the same number of elements."
    
    # 创建一个图形实例
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(points.shape[0]):
        point = points[i]
        vector_diff = vector_diffs[i]
        # 终点位置
        end_point = point + vector_diff
        # 在图中画出起点
        ax.scatter(point[0], point[1], point[2], color='r', label='Start Point' if i == 0 else "")
        # 使用 quiver 画出向量
        ax.quiver(point[0], point[1], point[2],
                  vector_diff[0], vector_diff[1], vector_diff[2],
                  length=1, normalize=True, color="g")
    if filename:
        fig.savefig(filename, bbox_inches='tight')

def plot_vector_pairs(points1, points2, filename=None):
    assert points1.shape == points2.shape, "两点序列的形状必须一致"
    
    # 创建一个图形实例
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制散点
    ax.scatter(points1[:, 0], points1[:, 1], points1[:, 2], color='r', label='Point Set 1')
    ax.scatter(points2[:, 0], points2[:, 1], points2[:, 2], color='b', label='Point Set 2')
    
    # 连接每对点
    for i in range(points1.shape[0]):
        ax.quiver(points1[i, 0], points1[i, 1], points1[i, 2],
                  points2[i, 0] - points1[i, 0], points2[i, 1] - points1[i, 1], points2[i, 2] - points1[i, 2],
                  length=1, normalize=False, color="g")
    
    if filename:
        fig.savefig(filename, bbox_inches='tight')