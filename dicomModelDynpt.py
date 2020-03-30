import vtk
import numpy as np
import pydicom as dicom
import platform
import os
import time
import vtk.util.numpy_support as vtknp
import glob
import re


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


def readFilesToDicomArray(path, listOfSeries):
    listOfVTKDataLists = []
    listOfPixelDims = []
    listOfPixelSpacings = []
    listOfMaxCounts = []
    listOfMatrices = []


    dictFilesDCM = {}

    for series in listOfSeries:  # für jeden Ordner
        for dirName, subdirList, fileList in os.walk(path + series):
            for filename in fileList:
                if ".dcm" in filename.lower():
                    actDs = dicom.read_file(os.path.join(dirName, filename))
                    pos = "{} {}".format(actDs.ImagePositionPatient, actDs.ImageOrientationPatient)

                    if (pos not in dictFilesDCM):
                        dictFilesDCM[pos] = {}
                    dictFilesDCM[pos][actDs.InstanceNumber] = os.path.join(dirName, filename)


    for actPos, actDict in dictFilesDCM.items():  # für jede Slice
        sortEntries = sorted(actDict)

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

                ConstPixelSpacing = (float(actDicom.PixelSpacing[0]),
                                     float(actDicom.PixelSpacing[1]),
                                     1.0)
                                     #float(actDicom.SliceThickness))

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

            ArrayDicom[actIndex, :, :] = actDicom.pixel_array
            actIndex += 1

        np.clip((resSlope * diffValGr / winWidth) * ArrayDicom + (((resIntercept - winCen) / winWidth + 0.5) * diffValGr + minValGr),
                   minValGr, maxValGr, out = ArrayDicom)

        VTK_dataList = []

        for actImage in range(len(ArrayDicom)):
            VTK_dataList.append(vtknp.numpy_to_vtk(ArrayDicom[actImage].ravel(),deep=True, array_type=vtk.VTK_FLOAT))

        listOfVTKDataLists.append(VTK_dataList)
        listOfMaxCounts.append(len(sortEntries))
        listOfPixelDims.append(ConstPixelDims)
        listOfPixelSpacings.append(ConstPixelSpacing)
        listOfMatrices.append(matrix)

    return (listOfVTKDataLists, listOfPixelDims, listOfPixelSpacings,
            listOfMaxCounts, listOfMatrices)


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


##########################################
##########################################


if platform.platform()[0] == "W":
    print("OS: win")
    pathIn = "c:/users/vch/desktop/Bredies/CASE01/"

else:
    print("OS: not win")
    pathIn = "/home/horakv/Desktop/Bredies/CASE01/"

##########################################
##########################################

seriesList = []

seriesList.append("segmentation/Visit_1___MRI_Data_and_Images_14d/B0553_90_MultiLabel_seg-1.3.6.1.4.1.16787.100.1.2.20170301.9093758712.1300")

###############################################################################
# Vorbereitung für Rendering
###############################################################################

ren = vtk.vtkRenderer()
#ren.SetBackground(0.8, 0.8, 0.8)
ren.SetBackground(1, 1, 1)

renWin = vtk.vtkRenderWindow()
renWin.SetSize(1000, 1000)

renWin.AddRenderer(ren)

iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)


###############################################################################
# Struktur des Modells einlesen
###############################################################################

filenames = glob.glob(pathIn + 'mesh/*.vtk')

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

mActor.GetProperty().SetOpacity(0.2)

if scalarRange == (0.0, 1.0):
    mMapper.ScalarVisibilityOff()
    mActor.GetProperty().SetColor(1,0,0)

ren.AddActor(mActor)


# =============================================================================
# (axes, widget) = showAxes()
# =============================================================================


###############################################################################
# Dicoms einlesen
###############################################################################

t0 = time.time()

(listOfVTKDataLists, listOfPixelDims, listOfPixelSpacings, listOfMaxCounts,
 listOfMatrices) = readFilesToDicomArray(pathIn, seriesList)

numImages = len(listOfVTKDataLists)

t1 = time.time()
print("Zeit:", t1-t0)


###############################################################################
# Dicoms für Visualisierung vorbereiten
###############################################################################

lookupTable = vtk.vtkLookupTable()
lookupTable.SetNumberOfTableValues(256)
lookupTable.SetRange(0.0, 255.0)
for j in range(256):
    lookupTable.SetTableValue(j, j/255.0, j/255.0, j/255.0, min(j/255.0*5, 1.0))

lookupTable.Build()

images = []

for actImage in range(numImages):  # für jede Slice
    countList.append(-1)

    image = vtk.vtkImageData()

    image.SetDimensions(listOfPixelDims[actImage][1], listOfPixelDims[actImage][2], 1)
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
# Rendering-Rest & beenden
###############################################################################

iren.Initialize()

renWin.Render()
print("Start")
iren.Start()

if platform.platform()[0] != "W":
    close_window(iren)
    del renWin, iren
