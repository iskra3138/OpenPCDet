import laspy
import numpy as np
import open3d

index = 99 #474

las_path = '/nas2/YJ/git/tree_data_raw/pcdet/data/tree/training/velodyne/T{}.las'.format(index)

lasfile = laspy.file.File(las_path, mode="r")
las_points = np.vstack((lasfile.x -30.0 , lasfile.y -30.0
                                 , lasfile.z-300.0)).transpose()
label_file = '/nas2/YJ/git/tree_data_raw/pcdet/data/tree/training/labels/{}.txt'.format(index)

'''
objects = []
with open(label_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        label = line.strip().split(' ')
        h = float(label[8])
        l = float(label[9])
        w = float(label[10])
        loc = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
        gt = [loc[0]-30.0, loc[1]-30.0, loc[2]-300.0, l, w, h, 0]
        objects.append(gt)

'''

## Save 3D as an image
vis = open3d.visualization.Visualizer()
vis.create_window(visible=False)

vis.get_render_option().point_size = 1.0
vis.get_render_option().background_color = np.zeros(3)

axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
vis.add_geometry(axis_pcd)

pts = open3d.geometry.PointCloud()
pts.points = open3d.utility.Vector3dVector(las_points)

vis.add_geometry(pts)
pts.colors = open3d.utility.Vector3dVector(np.ones((las_points.shape[0], 3)))

#### boxing ####
'''
for gt_boxes in objects :
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color((0,1,0))
    vis.add_geometry(line_set)
'''
#vis.update_renderer()
vis.run()
#vis.capture_screen_image('/nas2/YJ/test.png')
vis.destroy_window()
