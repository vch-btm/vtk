import vtk
import vtk.util.numpy_support as numpy_support
import numpy as np
import pydicom as dicom
import platform
import os
import vtk.util.numpy_support as vtknp
import re
import glob


minValGr = 0.0  # Skalierung der Grauwerte
maxValGr = 255.0
diffValGr = maxValGr - minValGr

countVol = -1
count = -1

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def close_window(iren):
    render_window = iren.GetRenderWindow()
    render_window.Finalize()
    iren.TerminateApp()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def showAxes():
    axes = vtk.vtkAxesActor()
    widget = vtk.vtkOrientationMarkerWidget()
    widget.SetOrientationMarker(axes)
    widget.SetInteractor(iren)
    widget.SetEnabled(1)
    widget.InteractiveOn()

    return(axes, widget)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def timer_callback(obj, event):
    global count, countVol

    ren.RemoveVolume(listOfVTKVolumes[countVol])
    countVol = (countVol + 1) % numTimeSteps
    ren.AddVolume(listOfVTKVolumes[countVol])

    count = (count + 20) % (maxCount - 1)
    #actMesh[...] = displacements[count]
    polydata.Modified()
    iren.GetRenderWindow().Render()



    #iren.Render()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def readDynpt():
    f = open(pathIn + "simulation/x.dynpt", 'rb')
    header = dict(re.findall(r"(\w*):(\w*)", f.read(1024).decode('utf-8')))

    shapeTest = [int(header['t']), int(header['x']), 3]

    data = np.fromfile(f, dtype=np.float32)

    if header['unites_x'] == "um":
        data /= 1000

    return(header, data.reshape(shapeTest))


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

if platform.platform()[0] == "W":
    print("OS: win")
    pathIn = "c:/users/vch/desktop/Bredies/CASE01/"


else:
    print("OS: not win")
    pathIn = "/home/horakv/Schreibtisch/Bredies/CASE01/"



seriesList = []

#seriesList.append("Bredies/cine/Visit_1___MRI_Data_and_Images_14d/ersteDicoms/")  # 25*16
#seriesList.append("Bredies/cine/Visit_1___MRI_Data_and_Images_14d/Smart-1.3.6.1.4.1.16787.100.1.2.20160613.9111835162.607/")  # 40
#seriesList.append("Bredies/cine/Visit_1___MRI_Data_and_Images_14d/Smart-1.3.6.1.4.1.16787.100.1.2.20160613.9111848390.608/")  # 40
#seriesList.append("Bredies/cine/Visit_1___MRI_Data_and_Images_14d/Smart-1.3.6.1.4.1.16787.100.1.2.20160613.9111901895.609/")  # 25*16
#seriesList.append("Bredies/cine/Visit_1___MRI_Data_and_Images_14d/Smart-1.3.6.1.4.1.16787.100.1.2.20160613.9112235900.610/")  # 40
#seriesList.append("Bredies/cine/Visit_1___MRI_Data_and_Images_14d/Smart-1.3.6.1.4.1.16787.100.1.2.20160613.9112254187.611/")  # 40
#seriesList.append("Bredies/cine/Visit_1___MRI_Data_and_Images_14d/Smart-1.3.6.1.4.1.16787.100.1.2.20160613.9112308236.612/")  # 25
#seriesList.append("Bredies/cine/Visit_1___MRI_Data_and_Images_14d/Smart-1.3.6.1.4.1.16787.100.1.2.20160613.9114136191.628/")  # 40
#seriesList.append("Bredies/cine/Visit_1___MRI_Data_and_Images_14d/Smart-1.3.6.1.4.1.16787.100.1.2.20160613.9114329783.631/")  # 40


seriesList.append("mri/Visit_1___MRI_Data_and_Images_14d/Smart-1.3.6.1.4.1.16787.100.1.2.20160613.9114423584.640/")  # 110

#seriesList.append("segmentation/Visit_1___MRI_Data_and_Images_14d/B0553_90_MultiLabel_seg-1.3.6.1.4.1.16787.100.1.2.20170301.9093758712.1300/")  # 110




numImages = 0

dictFilesDCM = {}
posList = []

for series in seriesList:  # für jeden Ordner
    for dirName, subdirList, fileList in os.walk(pathIn + series):
        for filename in fileList:
            if ".dcm" in filename.lower():
                actDs = dicom.read_file(os.path.join(dirName, filename))
                dictFilesDCM[actDs.InstanceNumber] = os.path.join(dirName, filename)

                pos = actDs.ImagePositionPatient

                if (pos not in posList):
                    posList.append(pos)

                numImages += 1

numSlices = len(posList)
numTimeSteps = int(numImages / numSlices)  # gleiche Anzahl angenommen


veryFirst = True

listOfVTKImageData = []

for actTime in range(numTimeSteps):  # für jeden Zeitschritt
    first = True

    for actSlice in range(numSlices):  # für jede Slice
        actDicom = dicom.read_file(dictFilesDCM[actSlice*numTimeSteps + actTime + 1])

        if first:
            first = False

            if veryFirst:
                veryFirst = False
                winCen = actDicom.WindowCenter
                winWidth = actDicom.WindowWidth
                resIntercept = actDicom.RescaleIntercept
                resSlope = actDicom.RescaleSlope

# Achtung: ConstPixelDims/Spacing evtl. Einträge 0 und 1 vertauscht
                ConstPixelDims = (int(actDicom.Rows),
                                  int(actDicom.Columns),
                                  numSlices)

                ConstPixelSpacing = (float(actDicom.PixelSpacing[0]),
                                     float(actDicom.PixelSpacing[1]),
                                     2.0)
                                     #float(actDicom.SliceThickness))

            ArrayDicom = np.zeros(ConstPixelDims, dtype=float, order='F')

        ArrayDicom[:, :, numSlices - actSlice - 1] = actDicom.pixel_array.transpose()

    # pdb.set_trace()
    np.clip((resSlope * diffValGr / winWidth) * ArrayDicom + (((resIntercept - winCen) / winWidth + 0.5) * diffValGr + minValGr), minValGr, maxValGr, out=ArrayDicom)
    ArrayDicom *= 2

    actVTKData = numpy_support.numpy_to_vtk(ArrayDicom.ravel(order='F'),
                                            deep=True, array_type=vtk.VTK_FLOAT)
    listOfVTKImageData.append(actVTKData)

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
    matrix.SetElement(i, 2, -zdir[i])
    matrix.SetElement(i, 3, position[i])


# # # # # # #

ren = vtk.vtkRenderer()
ren.ResetCamera()
ren.SetBackground(0.8, 0.8, 0.8)

listOfVTKVolumes = []

for i in range(len(listOfVTKImageData)):
    image = vtk.vtkImageData()
    image.AllocateScalars(vtk.VTK_FLOAT, 1)
    image.SetDimensions(ConstPixelDims)
    image.SetSpacing(ConstPixelSpacing)
    image.GetPointData().SetScalars(listOfVTKImageData[i])

    volumeMapper = vtk.vtkSmartVolumeMapper()
    volumeMapper.SetBlendModeToComposite()
    volumeMapper.SetRequestedRenderModeToGPU()
    volumeMapper.SetInputData(image)

    gradientOpacity = vtk.vtkPiecewiseFunction()
    # gradientOpacity.AddPoint(image.GetScalarRange()[0], 0.0)
    gradientOpacity.AddPoint(image.GetScalarRange()[1], 1.0)

    color = vtk.vtkColorTransferFunction()
    color.AddRGBPoint(image.GetScalarRange()[0], 0, 0, 0)
    color.AddRGBPoint(image.GetScalarRange()[1], 1, 1, 1)
    color.SetScaleToLinear()

    volumeProperty = vtk.vtkVolumeProperty()
    # volumeProperty.ShadeOn()
    volumeProperty.SetInterpolationTypeToLinear()

    volumeProperty.SetAmbient(0.1)
    volumeProperty.SetDiffuse(0.9)
    volumeProperty.SetSpecular(0.2)
    volumeProperty.SetSpecularPower(10.0)
    volumeProperty.SetColor(color)

    gradientOpacity.AddPoint(image.GetScalarRange()[0], 0.0)
    gradientOpacity.AddPoint(image.GetScalarRange()[1], 1.0)
    volumeProperty.SetGradientOpacity(gradientOpacity)

    opacityTransferFunction = vtk.vtkPiecewiseFunction()
    opacityTransferFunction.AddPoint(20, 0.0)
    opacityTransferFunction.AddPoint(255, 0.2)
    volumeProperty.SetScalarOpacity(opacityTransferFunction)

    volume = vtk.vtkVolume()
    volume.SetUserMatrix(matrix)
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    listOfVTKVolumes.append(volume)

ren.AddVolume(listOfVTKVolumes[0])

###########

bounds = volumeMapper.GetBounds()

cube = vtk.vtkCubeSource()
cube.SetBounds(bounds)

cubeMapper = vtk.vtkPolyDataMapper()
cubeMapper.SetInputConnection(cube.GetOutputPort())

cubeActor = vtk.vtkActor()
cubeActor.SetMapper(cubeMapper)
cubeActor.SetUserMatrix(matrix)
cubeActor.GetProperty().SetRepresentationToWireframe()
cubeActor.GetProperty().SetColor(0, 0, 0)

ren.AddActor(cubeActor)

###########

renWin = vtk.vtkRenderWindow()
renWin.SetSize(1000, 1000)

renWin.AddRenderer(ren)

iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

interactorStyle = vtk.vtkInteractorStyleTrackballCamera()
iren.SetInteractorStyle(interactorStyle)

# # # #

# filenames = glob.glob(path + 'mesh/*.vtk')
filenames = glob.glob(pathIn + 'mesh/*cutted.vtk')

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

(header, displacements) = readDynpt()

maxCount = len(displacements)

mMapper = vtk.vtkPolyDataMapper()
mMapper.SetInputData(polydata)
mMapper.SetScalarRange(scalarRange)



mActor = vtk.vtkActor()
mActor.SetMapper(mMapper)
mActor.SetPosition(getModelPosition())

mActor.GetProperty().SetOpacity(0.5)

if scalarRange == (0.0, 1.0):
    mMapper.ScalarVisibilityOff()
    mActor.GetProperty().SetColor(1,0,0)

ren.AddActor(mActor)


###########

points = vtk.vtkPoints()

pColors = vtk.vtkUnsignedCharArray()
pColors.SetNumberOfComponents(3)
pColors.SetName("Colors")

div = int(255/numSlices)

posList = sorted(posList)

for i in range(numSlices):
    points.InsertNextPoint(posList[i])
    #pColors.InsertNextTypedTuple([div * i, 0, div * (numSlices - i)])
    pColors.InsertNextTuple([div * i, 0, div * (numSlices - i)])

pointsPolydata = vtk.vtkPolyData()
pointsPolydata.SetPoints(points)

vertexFilter = vtk.vtkVertexGlyphFilter()
vertexFilter.SetInputData(pointsPolydata)
vertexFilter.Update()

pPolydata = vtk.vtkPolyData()
pPolydata.ShallowCopy(vertexFilter.GetOutput())

pPolydata.GetPointData().SetScalars(pColors)

pMapper = vtk.vtkPolyDataMapper()
pMapper.SetInputData(pPolydata)

pActor = vtk.vtkActor()
pActor.SetMapper(pMapper)
pActor.GetProperty().SetPointSize(5)

ren.AddActor(pActor)

###########

# =============================================================================
# (axes, widget) = showAxes()
# =============================================================================

iren.Initialize()
iren.AddObserver('TimerEvent', timer_callback)
iren.CreateRepeatingTimer(1)

renWin.Render()
print("Start")
iren.Start()

if platform.platform()[0] != "W":
    close_window(iren)
    del renWin, iren
