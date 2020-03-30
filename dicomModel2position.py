import vtk
import numpy as np
import pydicom as dicom
import platform
import os
import time
import vtk.util.numpy_support as vtknp
import glob
import re


withCushion = not True
showOpac = not True
showWireframe = not True

minValGr = 0.0  # Skalierung der Grauwerte
maxValGr = 255.0
diffValGr = maxValGr - minValGr

countList = []
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
    global updateVectors, numTimeSteps, numSlices, count, minDiffAtPos

    count = (count + 1) % numTimeSteps

    for actImage in range(numSlices):
        images[actImage].GetPointData().SetScalars(listOfVTKDataLists[actImage][updateVectors[actImage][count]])

    actMesh[...] = displacements[timeVectors[minDiffAtPos][updateVectors[minDiffAtPos][count]]]

    polydata.GetPoints().GetData().Modified()
    iren.GetRenderWindow().Render()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def calcUpdateVector(timeVectors, pos):
    numSlices = len(timeVectors)
    numTimeSteps = len(timeVectors[pos])

    updateVectors = np.ndarray((numSlices, numTimeSteps), int)

    for i in range(numSlices):
        lowerIndex = 0
        for j in range(numTimeSteps):
            needVal = True
            if i == pos:
                updateVectors[i][j] = j
                continue
            else:
                actTime = timeVectors[pos][j]
                upperIndex = len(timeVectors[i])

                lastDiff = float('Inf')

                for k in range(lowerIndex, upperIndex):
                    actDiff = int(abs(actTime - timeVectors[i][k]))

                    if actDiff == 0:
                        updateVectors[i][j] = k
                        needVal = False
                        break

                    if actDiff < lastDiff:
                        lastDiff = actDiff
                    else:
                        updateVectors[i][j] = k - 1
                        needVal = False
                        break

                    lowerIndex = k

            if needVal:
                updateVectors[i][j] = k

    return updateVectors

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def readFilesToDicomArray(path, listOfSeries):
    listOfDicomArrays = []
    listOfPixelDims = []
    listOfPixelSpacings = []
    listOfPlaneShapes = []
    listOfMaxCounts = []
    listOfMatrices = []


    dictFilesDCM = {}

    for series in listOfSeries:  # für jeden Ordner
        for dirName, subdirList, fileList in os.walk(path + series):
            for filename in fileList:
                if ".dcm" in filename.lower():
                    actDs = dicom.read_file(os.path.join(dirName, filename))
                    pos = "{}{}".format(actDs.ImagePositionPatient, actDs.ImageOrientationPatient)

                    if (pos not in dictFilesDCM):
                        dictFilesDCM[pos] = {}
                    dictFilesDCM[pos][actDs.InstanceNumber] = os.path.join(dirName, filename)

    minDiffAtPos = -1
    minDiff = float('Inf')

    timeVectors = []


    for actPos, actDict in dictFilesDCM.items():  # für jede Slice
        sortEntries = sorted(actDict)

        actTimeVector = []
        timeVectors.append(actTimeVector)

        first = True
        actIndex = 0

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

                ArrayDicom = np.zeros(ConstPixelDims, dtype = float)

            actTimeVector.append(int(actDicom.TriggerTime))

            ArrayDicom[actIndex, :, :] = actDicom.pixel_array
            actIndex += 1

        np.clip(resSlope * diffValGr / (winWidth - 1) * ArrayDicom + ((resIntercept - winCen) / (winWidth - 1) + 0.5) * diffValGr + minValGr,
                   minValGr, maxValGr, out = ArrayDicom)

        listOfMaxCounts.append(len(sortEntries))
        listOfDicomArrays.append(ArrayDicom)
        listOfPixelDims.append(ConstPixelDims)
        listOfPixelSpacings.append(ConstPixelSpacing)
        listOfPlaneShapes.append(planeShape)
        listOfMatrices.append(matrix)


    for i in range(len(timeVectors)):
        actTimeVector = timeVectors[i]
        factor = 800.0 / actTimeVector[-1]

        for j in range(len(actTimeVector)):
            actTimeVector[j] = int(factor * actTimeVector[j])

        if len(actTimeVector) > 0:
            tempDiff = actTimeVector[1] - actTimeVector[0]
            if tempDiff < minDiff:
                minDiff = tempDiff
                minDiffAtPos = i

    return (listOfDicomArrays, listOfPixelDims, listOfPixelSpacings,
            listOfPlaneShapes, listOfMaxCounts, listOfMatrices, minDiffAtPos,
            timeVectors)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def getAllVTKDataLists(listOfDicomArrays):

    resultList = []

    for ArrayDicom in listOfDicomArrays:
        VTK_dataList = []

        for actImage in range(len(ArrayDicom)):
            VTK_dataList.append(vtknp.numpy_to_vtk(ArrayDicom[actImage].ravel(),deep=True, array_type=vtk.VTK_FLOAT))

        resultList.append(VTK_dataList)

    return resultList


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


txtCase = "01"

if platform.platform()[0] == "W":
    print("OS: win")
    pathIn = "c:/users/vch/desktop/Bredies/CASE{}/".format(txtCase)

else:
    print("OS: not win")
    pathIn = "/home/horakv/Desktop/Bredies/CASE{}/".format(txtCase)


seriesList = []

seriesList.append("cine/Visit_1___MRI_Data_and_Images_14d/Smart-1.3.6.1.4.1.16787.100.1.2.20160613.9111835162.607/")  # 40
#seriesList.append("cine/Visit_1___MRI_Data_and_Images_14d/Smart-1.3.6.1.4.1.16787.100.1.2.20160613.9111848390.608/")  # 40
#seriesList.append("cine/Visit_1___MRI_Data_and_Images_14d/Smart-1.3.6.1.4.1.16787.100.1.2.20160613.9111901895.609/")  # 25*16
#seriesList.append("cine/Visit_1___MRI_Data_and_Images_14d/Smart-1.3.6.1.4.1.16787.100.1.2.20160613.9112235900.610/")  # 40
#seriesList.append("cine/Visit_1___MRI_Data_and_Images_14d/Smart-1.3.6.1.4.1.16787.100.1.2.20160613.9112254187.611/")  # 40
#seriesList.append("cine/Visit_1___MRI_Data_and_Images_14d/Smart-1.3.6.1.4.1.16787.100.1.2.20160613.9112308236.612/")  # 25
#seriesList.append("cine/Visit_1___MRI_Data_and_Images_14d/Smart-1.3.6.1.4.1.16787.100.1.2.20160613.9114136191.628/")  # 40
#seriesList.append("cine/Visit_1___MRI_Data_and_Images_14d/Smart-1.3.6.1.4.1.16787.100.1.2.20160613.9114329783.631/")  # 40

#t0 = time.time()

(listOfDicomArrays, listOfPixelDims, listOfPixelSpacings,
 listOfPlaneShapes, listOfMaxCounts, listOfMatrices, minDiffAtPos,
 timeVectors) = readFilesToDicomArray(pathIn, seriesList)

#t1 = time.time()

#print("Zeit:", t1-t0)

updateVectors = calcUpdateVector(timeVectors, minDiffAtPos)

numSlices = len(timeVectors)
numTimeSteps = len(timeVectors[minDiffAtPos])

numImages = len(listOfDicomArrays)

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

#############################
# =============================================================================
# axes = vtk.vtkAxesActor()
# widget = vtk.vtkOrientationMarkerWidget()
# widget.SetOrientationMarker(axes)
# widget.SetInteractor(iren)
# widget.SetEnabled( 1 )
# widget.InteractiveOn()
# =============================================================================
#############################


lookupTable = vtk.vtkLookupTable()
lookupTable.SetNumberOfTableValues(256)
lookupTable.SetRange(0.0, 255.0)
for j in range(256):
    lookupTable.SetTableValue(j, j/255.0, j/255.0, j/255.0, min(j/25.5, 1.0))
lookupTable.Build()

images = []

for actImage in range(numImages):
    image = vtk.vtkImageData()
    image.SetDimensions(listOfPlaneShapes[actImage])
    image.SetSpacing(listOfPixelSpacings[actImage][0], listOfPixelSpacings[actImage][1], 0.0)

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
    actor.SetUserMatrix(listOfMatrices[actImage])

    ren.AddActor(actor)



###############################################################################
# Struktur des Modells einlesen und Verschiebungen
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


if withCushion:
    (header, displacements) = readDynpt()
else:
    displacements = np.load(pathIn + 'simulation/disp_NC.npy')
    displacements /= 1000

maxCount = len(displacements)

mMapper = vtk.vtkPolyDataMapper()
mMapper.SetInputData(polydata)
mMapper.SetScalarRange(scalarRange)



mActor = vtk.vtkActor()
mActor.SetMapper(mMapper)
mActor.SetPosition(getModelPosition())

if showWireframe:
    showOpac = False
    mActor.GetProperty().SetRepresentationToWireframe()

if not showOpac:
    mActor.GetProperty().SetOpacity(0.2)

if scalarRange == (0.0, 1.0):
    mMapper.ScalarVisibilityOff()
    mActor.GetProperty().SetColor(1,0,0)

ren.AddActor(mActor)

###############################################################################

iren.Initialize()
iren.AddObserver('TimerEvent', timer_callback)
iren.CreateRepeatingTimer(10)


renWin.Render()
print("Start")
iren.Start()


if platform.platform()[0] != "W":
    close_window(iren)
    del renWin, iren
