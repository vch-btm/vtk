# coding=utf-8
import vtk
import numpy as np
import pydicom as dicom
import platform
import os
import vtk.util.numpy_support as vtknp
import glob
import re
import time

minValGr = 0.0  # Skalierung der Grauwerte
maxValGr = 255.0
diffValGr = maxValGr - minValGr

countList = []
count = -1

txtCase = "01"
pathToDisplay = 21
withCushion = not True
timeFactor = 6
numAvg = 1
withDTW = True


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def close_window(act_iren):
    render_window = act_iren.GetRenderWindow()
    render_window.Finalize()
    act_iren.TerminateApp()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def showAxes():
    axes = vtk.vtkAxesActor()
    widget = vtk.vtkOrientationMarkerWidget()
    widget.SetOrientationMarker(axes)
    widget.SetInteractor(iren)
    widget.SetEnabled(1)
    widget.InteractiveOn()

    return axes, widget


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def calcUpdateVector(vec, num_t):
    global timeFactor

    tempList = []

    numI = max(vec[:, 1]) + 1

    for m in range(numI):
        indices = [i for i, x in enumerate(vec[:, 1]) if x == m]

        # indices = [0, 1, 2, 797, 798, 799, 800]

        minEntry = min(indices)
        maxEntry = max(indices)

        if maxEntry - minEntry == num_t - 1:
            for n in range(num_t):
                if n == indices[n]:
                    indices[n] += (num_t - 1)
                else:
                    break

            minEntry = min(indices)
            maxEntry = max(indices)

        diff = (maxEntry - minEntry) / (timeFactor - 1)

        for n in range(timeFactor):
            tempList.append([m, round((minEntry + n * diff) % num_t)])

    return tempList


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def timer_callback(obj, event):
    global numTimeSteps, count, updateVec

    count = (count + 1) % len(updateVec)

    images[0].GetPointData().SetScalars(listOfVTKDataLists[actImage][updateVec[count][0]])

    allActors = ren.GetActors()

    for actActor in allActors:
        ren.RemoveActor(actActor)

    ren.AddActor(actor)

    tempCount = updateVec[count][1]

    ren.AddActor(allCuts[tempCount])
    ren.AddActor(allMMappers[tempCount])

    iren.GetRenderWindow().Render()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def readFilesToDicomArray(path_in, path_out):
    list_of_dicom_arrays = []
    list_of_pixel_dims = []
    list_of_pixel_spacings = []
    list_of_plane_shapes = []
    list_of_max_counts = []
    list_of_matrices = []

    dictFilesDCM = {}

    for dirName, subdirList, fileList in os.walk(path_in):
        for filename in fileList:
            if ".dcm" in filename.lower():
                actDs = dicom.read_file(os.path.join(dirName, filename))
                pos = "{}{}".format(actDs.ImagePositionPatient, actDs.ImageOrientationPatient)

                if pos not in dictFilesDCM:
                    dictFilesDCM[pos] = {}
                dictFilesDCM[pos][actDs.InstanceNumber] = os.path.join(dirName, filename)

    if withCushion:
        allPositionList = np.genfromtxt("{}withCushion/avg_{}/listInfoTemp.txt".format(path_out, numAvg), dtype='str',
                                        delimiter="  ")
    else:
        allPositionList = np.genfromtxt("{}noCushion/avg_{}/listInfoTemp.txt".format(path_out, numAvg), dtype='str',
                                        delimiter="  ")

    positionDict = {}

    for x, y in allPositionList:
        positionDict[y] = x

    for actPos, actDict in dictFilesDCM.items():  # für jede Slice
        actIndex = 0

        sortEntries = sorted(actDict)

        if not int(positionDict[actPos]) == pathToDisplay:
            continue

        first = True

        for actFile in sortEntries:  # für jedes einzelne Bild

            actDicom = dicom.read_file(actDict[actFile])

            if first:  # organisiere Metadaten + ArrayDicom anlegen
                first = False

                winCen = actDicom.WindowCenter
                winWidth = actDicom.WindowWidth
                resIntercept = actDicom.RescaleIntercept
                resSlope = actDicom.RescaleSlope

                ConstPixelDims = (len(sortEntries),
                                  int(actDicom.Rows),
                                  int(actDicom.Columns))

                planeShape = (int(actDicom.Rows), int(actDicom.Columns), 1)

                ConstPixelSpacing = (float(actDicom.PixelSpacing[0]),
                                     float(actDicom.PixelSpacing[1]),
                                     float(actDicom.SliceThickness))

                position = actDicom.ImagePositionPatient
                orientation = actDicom.ImageOrientationPatient

                xdir = orientation[0:3]
                ydir = orientation[3:6]
                zdir = [0.0, 0.0, 0.0]

                vtk.vtkMath.Cross(xdir, ydir, zdir)

                matrix = vtk.vtkMatrix4x4()

                for i in range(3):
                    matrix.SetElement(i, 0, xdir[i])
                    matrix.SetElement(i, 1, ydir[i])
                    matrix.SetElement(i, 2, zdir[i])
                    matrix.SetElement(i, 3, position[i])

                ArrayDicom = np.zeros(ConstPixelDims, dtype=float)

            ArrayDicom[actIndex, :, :] = actDicom.pixel_array
            actIndex += 1

        np.clip(resSlope * diffValGr / (winWidth - 1) * ArrayDicom + (
                (resIntercept - winCen) / (winWidth - 1) + 0.5) * diffValGr + minValGr,
                minValGr, maxValGr, out=ArrayDicom)

        list_of_max_counts.append(len(sortEntries))
        list_of_dicom_arrays.append(ArrayDicom)
        list_of_pixel_dims.append(ConstPixelDims)
        list_of_pixel_spacings.append(ConstPixelSpacing)
        list_of_plane_shapes.append(planeShape)
        list_of_matrices.append(matrix)

    return (list_of_dicom_arrays, list_of_pixel_dims, list_of_pixel_spacings,
            list_of_plane_shapes, list_of_max_counts, list_of_matrices)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def getAllVTKDataLists(list_of_dicom_arrays):
    resultList = []

    for array_dicom in list_of_dicom_arrays:
        VTK_dataList = []

        for act_image in range(len(array_dicom)):
            VTK_dataList.append(vtknp.numpy_to_vtk(array_dicom[act_image].ravel(), deep=True, array_type=vtk.VTK_FLOAT))

        resultList.append(VTK_dataList)

    return resultList


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def readDynpt():
    global pathIn, header

    f = open(pathIn + "simulation/x.dynpt", 'rb')
    header = dict(re.findall(r"(\w*):(\w*)", f.read(1024).decode('utf-8')))

    shapeTest = [int(header['t']), int(header['x']), 3]

    data = np.fromfile(f, dtype=np.float32)

    if header['unites_x'] == "um":
        data /= 1000

    return header, data.reshape(shapeTest)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def getModelPosition():
    minZPos = float("inf")

    for dirName, subdirList, fileList in os.walk(pathIn + "segmentation"):
        for filename in fileList:
            if ".dcm" in filename.lower():
                actDs = dicom.read_file(os.path.join(dirName, filename))
                actZPos = actDs.ImagePositionPatient[2]

                if actZPos < minZPos:
                    minZPos = actZPos

    return [actDs.ImagePositionPatient[0], actDs.ImagePositionPatient[1], minZPos]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

start = time.time()

if platform.platform()[0] == "W":
    print("OS: win")
    pathIn = "c:/users/vch/desktop/Bredies/CASE{}/".format(txtCase)
    pathOut = "c:/users/vch/desktop/results/CASE{}/".format(txtCase)
else:
    print("OS: not win")
    pathIn = "/home/horakv/Desktop/Bredies/CASE{}/".format(txtCase)
    pathOut = "/home/horakv/Desktop/results/CASE{}/".format(txtCase)

(listOfDicomArrays, listOfPixelDims, listOfPixelSpacings,
 listOfPlaneShapes, listOfMaxCounts, listOfMatrices) = readFilesToDicomArray(pathIn + "cine/", pathOut)

if withDTW:
    if withCushion:
        actVec = np.loadtxt("{}withCushion/avg_{}/path_{}.txt".format(pathOut, numAvg, pathToDisplay), dtype=int)
    else:
        actVec = np.loadtxt("{}noCushion/avg_{}/path_{}.txt".format(pathOut, numAvg, pathToDisplay), dtype=int)
else:
    if withCushion:
        actVec = np.loadtxt("{}withCushion/avg_{}/path_{}_sFMM.txt".format(pathOut, numAvg, pathToDisplay), dtype=int)
    else:
        actVec = np.loadtxt("{}noCushion/avg_{}/path_{}_sFMM.txt".format(pathOut, numAvg, pathToDisplay), dtype=int)

numImages = len(listOfDicomArrays)

numTimeSteps = len(actVec)

updateVec = calcUpdateVector(actVec, numTimeSteps)

###############################
# place for data manipulation #
###############################


listOfVTKDataLists = getAllVTKDataLists(listOfDicomArrays)

#########################################################

ren = vtk.vtkRenderer()
ren.SetBackground(0.8, 0.8, 0.8)

renWin = vtk.vtkRenderWindow()
renWin.SetSize(1000, 1000)

renWin.AddRenderer(ren)

iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

lookupTable = vtk.vtkLookupTable()
lookupTable.SetNumberOfTableValues(256)
lookupTable.SetRange(0.0, 255.0)
for j in range(256):
    lookupTable.SetTableValue(j, j / 255.0, j / 255.0, j / 255.0, min(j / 25.5, 1.0))
lookupTable.Build()

images = []

for actImage in range(numImages):
    image = vtk.vtkImageData()
    image.SetDimensions(listOfPlaneShapes[actImage])
    image.AllocateScalars(vtk.VTK_FLOAT, 1)
    image.GetPointData().SetScalars(listOfVTKDataLists[actImage][0])

    images.append(image)

    mapTransparency = vtk.vtkImageMapToColors()
    mapTransparency.SetLookupTable(lookupTable)
    mapTransparency.PassAlphaToOutputOn()
    mapTransparency.SetInputData(image)

    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(mapTransparency.GetOutputPort())
    mapper.SetColorModeToDirectScalars()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetInterpolationToFlat()
    actor.GetProperty().ShadingOff()
    actor.GetProperty().LightingOff()
    # actor.SetUserMatrix(listOfMatrices[actImage])

    ren.AddActor(actor)

###############################################################################
# Struktur des Modells einlesen
###############################################################################

if withCushion:
    filenames = glob.glob(pathIn + 'mesh/*.vtk')
else:
    filenames = glob.glob(pathIn + 'simulation/vtkNC/*.vtk')

reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName(filenames[0])

geometryFilter = vtk.vtkGeometryFilter()
geometryFilter.SetInputConnection(reader.GetOutputPort())
geometryFilter.Update()

polydata = geometryFilter.GetOutput()
scalarRange = polydata.GetScalarRange()

actMesh = vtknp.vtk_to_numpy(polydata.GetPoints().GetData())
actMesh /= 1000  # Daten sind in um statt in mm gegeben -> Korrektur

###############################################################################
# Verschiebungen vorbereiten und Visualisierung
###############################################################################

if withCushion:
    (header, displacements) = readDynpt()
else:
    displacements = np.load(pathIn + 'simulation/disp_NC.npy')
    displacements /= 1000

maxCount = len(displacements)

mPosition = getModelPosition()
helpMatrix = vtk.vtkMatrix4x4()
helpMatrix.DeepCopy(listOfMatrices[0])
helpMatrix.Invert()

allCuts = []
allMMappers = []

for actT in range(maxCount):
    actMesh[...] = displacements[actT]

    translation = vtk.vtkTransform()
    translation.SetMatrix(helpMatrix)
    translation.Translate(mPosition)

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputConnection(geometryFilter.GetOutputPort())
    transformFilter.SetTransform(translation)
    transformFilter.Update()

    polydata = transformFilter.GetOutput()

    scalingVec = [listOfPixelSpacings[0][0], listOfPixelSpacings[0][1],
                  (listOfPixelSpacings[0][0] + listOfPixelSpacings[0][1]) / 2]

    polydata.GetPoints().SetData(
        vtknp.numpy_to_vtk(np.divide(polydata.GetPoints().GetData(), scalingVec), array_type=vtk.VTK_FLOAT))

    mMapper = vtk.vtkPolyDataMapper()
    mMapper.SetInputData(polydata)
    mMapper.SetScalarRange(scalarRange)

    mActor = vtk.vtkActor()
    mActor.SetMapper(mMapper)
    mActor.GetProperty().SetOpacity(0.2)

    # mMapper.ScalarVisibilityOff()
    # mActor.GetProperty().SetColor(1, 0, 0)

    allMMappers.append(mActor)

    plane = vtk.vtkPlane()

    cutEdges = vtk.vtkCutter()
    cutEdges.SetInputData(polydata)
    cutEdges.SetCutFunction(plane)
    cutEdges.GenerateCutScalarsOn()

    cutStrips = vtk.vtkStripper()
    cutStrips.SetInputConnection(cutEdges.GetOutputPort())
    cutStrips.Update()
    cutPoly = vtk.vtkPolyData()
    cutPoly.SetPoints(cutStrips.GetOutput().GetPoints())
    cutPoly.SetPolys(cutStrips.GetOutput().GetLines())

    cutMapper = vtk.vtkPolyDataMapper()
    cutMapper.SetInputData(cutPoly)
    cutActor = vtk.vtkActor()
    cutActor.SetMapper(cutMapper)
    cutActor.GetProperty().SetRepresentationToWireframe()
    cutMapper.ScalarVisibilityOff()
    cutActor.GetProperty().SetColor(1, 0, 0)

    allCuts.append(cutActor)

###############################################################################

iren.Initialize()
iren.AddObserver('TimerEvent', timer_callback)
iren.CreateRepeatingTimer(1)

end = time.time()

print(end - start)

renWin.Render()
print("Start")
iren.Start()

if platform.platform()[0] != "W":
    close_window(iren)
    del renWin, iren
