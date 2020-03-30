import vtk
import vtk.util.numpy_support as vtknp
import numpy as np
import pydicom as dicom
import platform
import glob
import os
import re
from collections import defaultdict
import cairocffi as cairo


txtCase = "01"
withCushion = not True

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


def getTriggerTimes(folder, file):
    triggerTime = 0
    triggerTimeFirst = float("inf")
    triggerTimeLast = -float("inf")

    for dirName, subdirList, fileList in os.walk(pathIn + folder):
        for filename in fileList:
            if ".dcm" in filename.lower():
                actDs = dicom.read_file(os.path.join(dirName, filename))
                actTriggerTime = actDs.TriggerTime

                if filename == file:
                    triggerTime = actTriggerTime

                if actTriggerTime < triggerTimeFirst:
                    triggerTimeFirst = actTriggerTime

                if actTriggerTime > triggerTimeLast:
                    triggerTimeLast = actTriggerTime


    return triggerTime, triggerTimeFirst, triggerTimeLast


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


##########################################
##########################################

if platform.platform()[0] == "W":
    print("OS: win")
    pathIn = "c:/users/vch/desktop/Bredies/CASE{}/".format(txtCase)
else:
    print("OS: not win")
    pathIn = "/home/horakv/Desktop/Bredies/CASE{}/".format(txtCase)

##########################################
##########################################

if withCushion:
    filenames = glob.glob(pathIn + 'mesh/*.vtk')
else:
    filenames = glob.glob(pathIn + 'simulation/vtkNC/*.vtk')


#folder = "cine/Visit_1___MRI_Data_and_Images_14d/Smart-1.3.6.1.4.1.16787.100.1.2.20160613.9111901895.609/"
#file = "1.3.6.1.4.1.16787.100.1.3.20160613.9111938418.dcm"

folder = "cine/Visit_1___MRI_Data_and_Images_14d/Smart-1.3.6.1.4.1.16787.100.1.2.20160613.9111835162.607/"
file = "1.3.6.1.4.1.16787.100.1.3.20160613.9111835163.dcm"
#file = "1.3.6.1.4.1.16787.100.1.3.20160613.9111843145.dcm"

actTriggerTime, triggerTimeFirst, triggerTimeLast = getTriggerTimes(folder, file)

actDicom = dicom.read_file(pathIn + folder + file)

winCen = actDicom.WindowCenter
winWidth = actDicom.WindowWidth
resIntercept = actDicom.RescaleIntercept
resSlope = actDicom.RescaleSlope

pixelDims = (1, int(actDicom.Rows), int(actDicom.Columns))

planeShape = (int(actDicom.Rows), int(actDicom.Columns), 1)

pixelSpacing = (float(actDicom.PixelSpacing[0]), float(actDicom.PixelSpacing[1]), 1)

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

ArrayDicom = np.zeros(pixelDims, dtype = float)
ArrayDicom[:, :] = actDicom.pixel_array

np.clip((resSlope * diffValGr / winWidth) * ArrayDicom + (((resIntercept - winCen) / winWidth + 0.5) * diffValGr + minValGr), minValGr, maxValGr, out = ArrayDicom)

vtkData = vtknp.numpy_to_vtk(ArrayDicom.ravel(),deep=True, array_type=vtk.VTK_FLOAT)


###############################################################################
# Darstellung-Vorbereitung
###############################################################################

renDicom = vtk.vtkRenderer()
renWinDicom = vtk.vtkRenderWindow()
renWinDicom.AddRenderer(renDicom)
irenDicom = vtk.vtkRenderWindowInteractor()
irenDicom.SetRenderWindow(renWinDicom)


###############################################################################
# Dicom-Part 2
###############################################################################

lookupTable = vtk.vtkLookupTable()
lookupTable.SetNumberOfTableValues(256)
lookupTable.SetRange(0.0, 255.0)
for j in range(256):
    lookupTable.SetTableValue(j, j/255.0, j/255.0, j/255.0, min(j/255.0*5, 1.0))
    #lookupTable.SetTableValue(j, j/255.0, j/255.0, j/255.0, min(j/255.0*20, 1.0))  # für Modell
lookupTable.Build()

image = vtk.vtkImageData()
image.SetDimensions(planeShape)

image.AllocateScalars(vtk.VTK_FLOAT, 1)
image.GetPointData().SetScalars(vtkData)

mapTransparency = vtk.vtkImageMapToColors()
mapTransparency.SetLookupTable(lookupTable)
mapTransparency.PassAlphaToOutputOn()
mapTransparency.SetInputData(image)

mapper = vtk.vtkDataSetMapper()
#mapper.ImmediateModeRenderingOn()
mapper.SetInputConnection(mapTransparency.GetOutputPort())
mapper.SetColorModeToDirectScalars()

actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetInterpolationToFlat()
actor.GetProperty().ShadingOff()
actor.GetProperty().LightingOff()
actor.GetProperty().SetOpacity(1.0)

renDicom.AddActor(actor)


###############################################################################
# Modell-Erstellung und Positionierung
###############################################################################

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

mNumberTimeSteps = len(displacements)


if not triggerTimeLast == 0.0:
    actMesh[...] = displacements[round((actTriggerTime - triggerTimeFirst) * (mNumberTimeSteps - 1) / (triggerTimeLast - triggerTimeFirst))]

mPosition = getModelPosition()

polydata = geometryFilter.GetOutput()

helpMatrix = vtk.vtkMatrix4x4()
helpMatrix.DeepCopy(matrix)
helpMatrix.Invert()

translation = vtk.vtkTransform()
translation.SetMatrix(helpMatrix)
translation.Translate(mPosition)

transformFilter = vtk.vtkTransformPolyDataFilter()
transformFilter.SetInputConnection(geometryFilter.GetOutputPort())
transformFilter.SetTransform(translation)
transformFilter.Update()

polydata = transformFilter.GetOutput()

allPoints = polydata.GetPoints()
numPoints = allPoints.GetNumberOfPoints()

scalingVec = [pixelSpacing[0], pixelSpacing[1],
                          (pixelSpacing[0] + pixelSpacing[1]) / 2]

polydata.GetPoints().SetData(vtknp.numpy_to_vtk(np.divide(polydata.GetPoints().GetData(), scalingVec), array_type=vtk.VTK_FLOAT))


###############################################################################
# visuelle Kontrolle
###############################################################################

mMapper = vtk.vtkPolyDataMapper()
#mMapper.ImmediateModeRenderingOn()
mMapper.SetInputData(polydata)
mActor = vtk.vtkActor()
mActor.SetMapper(mMapper)
mActor.GetProperty().SetOpacity(0.15)

if scalarRange == (0.0, 1.0):
    mMapper.ScalarVisibilityOff()
    mActor.GetProperty().SetColor(1,0,0)
#renDicom.AddActor(mActor)


###############################################################################
# rechnische Vorbereitung auf Schnitt
###############################################################################

normalGenerator = vtk.vtkPolyDataNormals()
normalGenerator.SetInputData(polydata)
normalGenerator.ComputePointNormalsOn()
normalGenerator.ComputeCellNormalsOff()
normalGenerator.Update()
polydata = normalGenerator.GetOutput()

normalGenerator = vtk.vtkPolyDataNormals()
normalGenerator.SetInputData(polydata)
normalGenerator.ComputePointNormalsOff()
normalGenerator.ComputeCellNormalsOn()
normalGenerator.Update()
polydata = normalGenerator.GetOutput()


###############################################################################
# Schnitt und Erstellung geschl. Kurve(n)
###############################################################################

modelNormals = vtk.vtkPolyDataNormals()
modelNormals.SetInputData(polydata)

plane = vtk.vtkPlane()

cutEdges = vtk.vtkCutter()
cutEdges.SetInputConnection(modelNormals.GetOutputPort())
cutEdges.SetCutFunction(plane)
cutEdges.GenerateCutScalarsOn()
#cutEdges.SetValue(0, 0.5)

cutStrips = vtk.vtkStripper()
cutStrips.SetInputConnection(cutEdges.GetOutputPort())
cutStrips.Update()
cutPoly = vtk.vtkPolyData()
cutPoly.SetPoints(cutStrips.GetOutput().GetPoints())
cutPoly.SetPolys(cutStrips.GetOutput().GetLines())


###############################################################################
# even-odd-filling
###############################################################################

right = [planeShape[0], planeShape[1]]

cln = vtk.vtkCleanPolyData()
cln.SetInputData(cutPoly)
cln.SetInputConnection(cutEdges.GetOutputPort())
cln.Update()
pd = cln.GetOutput()
points = vtknp.vtk_to_numpy(pd.GetPoints().GetData())
lines = vtknp.vtk_to_numpy(pd.GetLines().GetData())
p0 = lines[1::3]
p1 = lines[2::3]
# build connection graph
pp = defaultdict(set)
for i in range(len(p0)):
    pp[p0[i]].add(p1[i])
    pp[p1[i]].add(p0[i])
neighbors = {p: len(q) for (p, q) in pp.items()}
starters = [p for (p, q) in neighbors.items() if np.mod(q, 2) == 1]
# traverse possible paths
contours = []
while (len(pp) > 0):
    curcontour = []
    if len(starters) > 0:
        curpoint = starters.pop()
    else:
        curpoint = list(pp.keys())[0]
    while curpoint in pp:
        curcontour.append(curpoint)
        nextpoint = pp[curpoint].pop()
        if (len(pp[curpoint]) == 0):
            pp.pop(curpoint)
        pp[nextpoint].remove(curpoint)
        if (len(pp[nextpoint]) == 0):
            pp.pop(nextpoint)
        curpoint = nextpoint
    if (len(curcontour) > 0):
        contours.append(curcontour)
        #print("%d contours, %d points remaining" % (len(contours), len(pp)))

surface = cairo.ImageSurface(cairo.FORMAT_A8, right[0], right[1])

context = cairo.Context(surface)
context.scale(1, 1)
context.set_line_width(1)
context.set_source_rgb(1, 1, 1)
context.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)

for c in contours:
    context.move_to(points[c[0], 0], points[c[0], 1])
    for p in c:
        context.line_to(points[p, 0], points[p, 1])
    context.close_path()

context.fill()

surface.write_to_png("prototype.png")


###############################################################################
# visuell schönere Schnittdarstellung
###############################################################################

connect = vtk.vtkPolyDataConnectivityFilter()
connect.SetInputData(cutPoly)
connect.SetExtractionModeToSpecifiedRegions()
connect.ColorRegionsOn()
connect.Update()

numRegions = connect.GetNumberOfExtractedRegions()

for i in range(numRegions):
    connect.InitializeSpecifiedRegionList()
    connect.AddSpecifiedRegion(i)
    connect.Modified()
    connect.Update()

    extractRegionData = vtk.vtkPolyData()
    extractRegionData.DeepCopy(connect.GetOutput())

    regionMapper = vtk.vtkPolyDataMapper()
    #regionMapper.ImmediateModeRenderingOn()
    regionMapper.SetInputData(extractRegionData)
    regionActor = vtk.vtkActor()
    regionActor.SetMapper(regionMapper)
    regionActor.GetProperty().SetRepresentationToWireframe()

    renDicom.AddActor(regionActor)

# =============================================================================
# axes = vtk.vtkAxesActor()
# widget = vtk.vtkOrientationMarkerWidget()
# widget.SetOrientationMarker(axes)
# widget.SetInteractor(irenDicom)
# widget.SetEnabled( 1 )
# widget.InteractiveOn()
# =============================================================================


winWidth = planeShape[0]
winHeight = planeShape[1]

renWinDicom.SetSize(winWidth, winHeight)

renDicom.SetBackground(1, 1, 1)

irenDicom.SetRenderWindow(renWinDicom)
renWinDicom.AddRenderer(renDicom)

renWinDicom.Render()
irenDicom.Start()


if platform.platform()[0] != "W":
    close_window(irenDicom)
