# coding=utf-8
import vtk
import vtk.util.numpy_support as vtknp
import numpy as np
import platform
import glob


txtCase = "01"

if platform.platform()[0] == "W":
    print("OS: win")
    pfad = "c:/users/vch/desktop/Bredies/CASE{}/simulation/vtkNC/".format(txtCase)
    pfadOut = "c:/users/vch/desktop/Bredies/CASE{}/simulation/".format(txtCase)
else:
    print("OS: not win")
    pfad = "/home/horakv/Desktop/Bredies/CASE{}/simulation/vtkNC/".format(txtCase)
    pfadOut = "/home/horakv/Desktop/Bredies/CASE{}/simulation/".format(txtCase)


filenames = glob.glob(pfad + '*.vtk')

reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName(filenames[0])

geometryFilter = vtk.vtkGeometryFilter()
geometryFilter.SetInputConnection(reader.GetOutputPort())
geometryFilter.Update()

polydata = geometryFilter.GetOutput()
scalarRange = polydata.GetScalarRange()

actMesh = vtknp.vtk_to_numpy(polydata.GetPoints().GetData())


displacements = []
substrings = filenames[0].split("_")


for i in range(len(filenames)):
    print("{}_{}_{}.vtk".format(substrings[0], substrings[1], i))
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName("{}_{}_{}.vtk".format(substrings[0], substrings[1], i))
    reader.Update()
    cur_heart = reader.GetOutput()

    displacements.append(actMesh + vtknp.vtk_to_numpy(cur_heart.GetPointData().GetVectors()))

np.save(pfadOut + 'disp_NC.npy', displacements)
