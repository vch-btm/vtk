import glob
import vtk
import vtk.util.numpy_support as vtknp
import platform
import os
import re
import numpy as np
import pydicom as dicom
from scipy import ndimage
from collections import defaultdict
import cairo
from PIL import ImageDraw, Image
import numpy.ma as ma
import time
import shutil
import matplotlib.pyplot as plt


maskCorr = 5
pixelCorr = 2 * maskCorr + 5

withCushion = not True
txtCase = "01"


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def dtw(matrix):
    r, c = matrix.shape
    D0 = np.zeros((r + 1, c + 1))
    D0[0, 1:] = np.inf
    D0[1:, 0] = np.inf
    D1 = D0[1:, 1:] # view
    for i in range(r):
        for j in range(c):
            D1[i, j] = matrix[i, j]
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])

    p, q, path = _traceback(D0)

    return D1[-1, -1] / sum(D1.shape), C, D1, p, q, path


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def _traceback(D):
    i, j = np.array(D.shape) - 2

    p, q = [i], [j]
    path = []
    path.insert(0, (i, j))


    while ((i > 0) or (j > 0)):
        tb = np.argmin((D[i, j], D[i, j+1], D[i+1, j]))
        if (tb == 0):
            i -= 1
            j -= 1
        elif (tb == 1):
            i -= 1
        else: # (tb == 2):
            j -= 1

        path.insert(0, (i, j))
        p.insert(0, i)
        q.insert(0, j)

    return np.array(p), np.array(q), np.array(path)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def getBestPath(matrix):

    bestI = -1
    bestMat1 = 0
    bestMat2 = 0
    bestp = 0
    bestq = 0
    bestDist = np.inf
    bestPath = 0


    length = len(matrix[0])


    for i in range(1, length + 1):
        rolledMatrix = np.roll(matrix, i, axis=1)

        dist, mat1, mat2, p, q, path = dtw(rolledMatrix)

        if dist < bestDist:
            bestI = i
            bestMat1 = mat1
            bestMat2 = mat2
            bestp = p
            bestq = q
            bestDist = dist
            bestPath = path


    for i in range(len(bestPath)):
        bestPath[i, 1] = (bestPath[i, 1] - bestI + length) % length
        bestq[i] = (bestq[i] - bestI + length) % length

    return bestPath, bestDist, bestMat1, bestp, bestq


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

    fileList = glob.glob(pathIn + 'segmentation/*/*/*.dcm')

    for filename in fileList:
        actDicom = dicom.read_file(filename)
        actZPos = actDicom.ImagePositionPatient[2]

        if actZPos < minZPos:
            minZPos = actZPos

    return [actDicom.ImagePositionPatient[0],
            actDicom.ImagePositionPatient[1],
            minZPos]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


##########################################
##########################################

if platform.platform()[0] == "W":
    print("OS: win")
    pathIn = "c:/users/vch/desktop/Bredies/CASE{}/".format(txtCase)
    pathOut = "c:/users/vch/desktop/results/CASE{}/".format(txtCase)


else:
    print("OS: not win")
    pathIn = "/home/horakv/Desktop/Bredies/CASE{}/".format(txtCase)
    pathOut = "/home/horakv/Desktop/results/CASE{}/".format(txtCase)


if not os.path.exists(pathOut):
    os.makedirs(pathOut)

##########################################
##########################################

t0 = time.time()


###############################################################################
# Modell + Verschiebungen lesen
###############################################################################


withSquare = True

if not withSquare:
    pathOut = "/home/horakv/Desktop/resultsRoot/"

if withCushion:
    filenames = glob.glob(pathIn + 'mesh/*.vtk')
    pathOut += "withCushion/noAvg/"
else:
    filenames = glob.glob(pathIn + 'simulation/vtk_NC/*.vtk')
    pathOut += "noCushion/noAvg/"

if not os.path.exists(pathOut):
    os.makedirs(pathOut)

textfile = open(pathOut + "list.txt", "w")
textfile.write(pathIn + '\n')

reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName(filenames[0])

geometryFilter = vtk.vtkGeometryFilter()
geometryFilter.SetInputConnection(reader.GetOutputPort())
geometryFilter.Update()

mPolydata = geometryFilter.GetOutput()

mMesh = vtknp.vtk_to_numpy(mPolydata.GetPoints().GetData())
mMesh /= 1000  # Daten sind in um statt in mm gegeben . Korrektur

if withCushion:
    (header, displacements) = readDynpt()
else:
    displacements = np.load(pathIn + 'simulation/disp_NC.npy')
    displacements /= 1000

mNumberTimeSteps = len(displacements)
mPosition = getModelPosition()


#########################################
#########################################


dictFilesDCM = {}

#########################################
# for dirName, subdirList, fileList in os.walk(pathIn + "mri/"):
# for dirName, subdirList, fileList in os.walk(pathIn + "segmentation/"):
for dirName, subdirList, fileList in os.walk(pathIn + "cine/"):
    for filename in fileList:
        if ".dcm" in filename.lower():
            actDicom = dicom.read_file(os.path.join(dirName, filename))
            posOri = "{}{}".format(actDicom.ImagePositionPatient, actDicom.ImageOrientationPatient)

            if (posOri not in dictFilesDCM):
                dictFilesDCM[posOri] = {}
            dictFilesDCM[posOri][actDicom.InstanceNumber] = os.path.join(dirName, filename)
#########################################


# construct a circle for dilatation
x = range(-maskCorr, maskCorr + 1)
[X, Y] = np.meshgrid(x, x)
struct = (X * X + Y * Y <= maskCorr ** 2)


l2normSum = 0.0

actSlice = 0

textfileInfos = open(pathOut + "listInfo.txt", "w")

for actPos, actDict in dictFilesDCM.items():  # für jede Slice

    #if actSlice == -1:
    if not actSlice == 5:
        actSlice += 1
        continue

    sortEntries = sorted(actDict)

    textfileInfos.write("{}  {}\n".format(actSlice, actPos))

    pathOutTemp = "{}{}/".format(pathOut, actSlice)

    if not os.path.exists(pathOutTemp):
        os.makedirs(pathOutTemp)

    tempMatrix = np.zeros((mNumberTimeSteps, len(sortEntries)))

    first = True
    actIndex = 0

    seriesOfDicoms = []

    for actFile in sortEntries:  # für jedes einzelne Bild
        actDicom = dicom.read_file(actDict[actFile])

        if first:  # organisiere Metadaten
            first = False

            resIntercept = actDicom.RescaleIntercept
            resSlope = actDicom.RescaleSlope

            pixelDims = (len(sortEntries),
                         int(actDicom.Rows),
                         int(actDicom.Columns))

            try:
                pixelSpacing = (float(actDicom.PixelSpacing[0]),
                                float(actDicom.PixelSpacing[1]),
                                float(actDicom.SliceThickness))
            except AttributeError:
                pixelSpacing = (float(actDicom.PixelSpacing[0]),
                                float(actDicom.PixelSpacing[1]),
                                1.0)

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

            scalingVec = [pixelSpacing[0], pixelSpacing[1],
                          (pixelSpacing[0] + pixelSpacing[1]) / 2]

            matrix.Invert()

            translation = vtk.vtkTransform()
            translation.SetMatrix(matrix)
            translation.Translate(mPosition)

            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetTransform(translation)


        arrayDicom = np.zeros((actDicom.Rows, actDicom.Columns), dtype=float)

        arrayDicom[:, :] = actDicom.pixel_array
        arrayDicom = resSlope * arrayDicom + resIntercept

        seriesOfDicoms.append((actDicom, arrayDicom))

    allContours = []
    allPoints = []
    allSplitInfos = []

    yMin = np.inf
    yMax = -np.inf

    xMin = np.inf
    xMax = -np.inf

    for t in range(mNumberTimeSteps):
        polydata = vtk.vtkPolyData()
        polydata.DeepCopy(mPolydata)

        actMesh = vtknp.vtk_to_numpy(polydata.GetPoints().GetData())
        actMesh[...] = displacements[t]

        polydata.Modified()

        transformFilter.SetInputData(polydata)
        transformFilter.Update()

        polydata = transformFilter.GetOutput()

        polydata.GetPoints().SetData(vtknp.numpy_to_vtk(np.divide(polydata.GetPoints().GetData(), scalingVec),
                                                        array_type=vtk.VTK_FLOAT))  # dividiere alle punkte durch scalingVec

        #######################################################################
        # finding contours
        #######################################################################

        plane = vtk.vtkPlane()

        cutter = vtk.vtkCutter()
        cutter.SetInputData(polydata)
        cutter.SetCutFunction(plane)
        cutter.GenerateCutScalarsOn()

        right = [pixelDims[1], pixelDims[2]]

        cln = vtk.vtkCleanPolyData()
        cln.SetInputConnection(cutter.GetOutputPort())
        cln.Update()

        pd = cln.GetOutput()

        splitInfo = re.split(r'[/\\]', actDict[actFile])

        needBreak = False

        if pd.GetPoints() is None:
            needBreak = True

            try:
                actTriggerTime = actDicom.TriggerTime
            except AttributeError:
                actTriggerTime = 0

            textfile.write(
                "{} {}/{}/{} {} {}\n".format(float("inf"), splitInfo[-4], splitInfo[-3], splitInfo[-2],
                                             actTriggerTime, t))
            break


        cutBounds = pd.GetBounds()

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
        # traverse possible pathIns
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

        allContours.append(contours)
        allPoints.append(points)
        allSplitInfos.append(splitInfo)

        #######################################################################
        # finding cut window
        #######################################################################

        yMin = int(min(int(max(0, cutBounds[0] - pixelCorr)), yMin))
        yMax = int(max(int(min(right[0], cutBounds[1] + pixelCorr)), yMax))

        xMin = int(min(int(max(0, cutBounds[2] - pixelCorr)), xMin))
        xMax = int(max(int(min(right[1], cutBounds[3] + pixelCorr)), xMax))

    for t in range(mNumberTimeSteps):
        actNum = 0

        if needBreak:
            break


        for actDicom, arrayDicom in seriesOfDicoms:  # für jedes einzelne Bild
            dicomCut = arrayDicom[xMin:xMax, yMin:yMax]

            try:
                actTriggerTime = actDicom.TriggerTime
            except AttributeError:
                actTriggerTime = 0


            #######################################################################
            # filling contours
            #######################################################################

            mData = np.zeros((right[0], right[1]), dtype=np.uint8)


            mSurface = cairo.ImageSurface.create_for_data(mData, cairo.FORMAT_A8, right[0], right[1])
            mContext = cairo.Context(mSurface)
            mContext.scale(1, 1)
            mContext.set_line_width(1)
            mContext.set_source_rgb(1, 1, 1)

            for c in allContours[t]:
                mContext.move_to(allPoints[t][c[0], 0], allPoints[t][c[0], 1])

                for p in c:
                    mContext.line_to(allPoints[t][p, 0], allPoints[t][p, 1])

                mContext.close_path()
                mContext.fill()

            mask = ndimage.binary_dilation(mData, structure=struct)[xMin:xMax, yMin:yMax]

            numPixels = np.sum(mask)

            listOfCutData = []

            for c in allContours[t]:
                data = np.zeros((right[0], right[1]), dtype=np.uint8)

                surface = cairo.ImageSurface.create_for_data(data, cairo.FORMAT_A8, right[0], right[1])
                context = cairo.Context(surface)
                context.scale(1, 1)
                context.set_line_width(1)
                context.set_source_rgb(1, 1, 1)

                context.move_to(allPoints[t][c[0], 0], allPoints[t][c[0], 1])

                for p in c:
                    context.line_to(allPoints[t][p, 0], allPoints[t][p, 1])
                context.close_path()

                context.fill()

                tempCut = data[xMin:xMax, yMin:yMax]

                if not np.sum(tempCut) == 0.0:
                    listOfCutData.append(tempCut/255)

            listOfCutData.append(np.ones(dicomCut.shape))  # weißes Grundbild

            numContours = len(listOfCutData)

            dicomCut = ma.masked_array(dicomCut, ~mask)

            for i in range(numContours):
                listOfCutData[i] = ma.masked_array(listOfCutData[i], ~mask)

            matrixP = np.zeros((numContours, numContours))
            vecD = np.zeros(numContours)

            for i in range(numContours):
                for j in range(numContours):
                    test = listOfCutData[i] * listOfCutData[j]
                    test2 = np.sum(test)

                    matrixP[i][j] = np.sum(listOfCutData[i] * listOfCutData[j])
                vecD[i] = np.sum(listOfCutData[i] * dicomCut)

            vecC = np.linalg.solve(matrixP, vecD)

            modelCut = ma.masked_array(np.zeros(dicomCut.shape), ~mask)

            for actContour in range(numContours):
                modelCut += listOfCutData[actContour]*vecC[actContour]

            #######################################################################
            # calc norm, prepare and save output
            #######################################################################

            l2norm = np.sqrt(np.sum(np.square(dicomCut - modelCut) / numPixels))
            l2normSum += l2norm**2

            textfile.write("{} {}/{}/{} {} {}\n".format(l2norm, allSplitInfos[t][-4],
                                                        allSplitInfos[t][-3], allSplitInfos[t][-2],
                                                        actTriggerTime, t))

            valToSave = 0

            if withSquare:
                valToSave = l2norm**2
            else:
                valToSave = l2norm


            tempMatrix[t][actNum] = valToSave

            maxValArrayDicom = arrayDicom.max()

            combArray = np.concatenate((modelCut, dicomCut), axis = 1)
            imgHeight, imgWidth = combArray.shape
            img = Image.fromarray((255 * np.concatenate((combArray, np.zeros((10, imgWidth)))) / maxValArrayDicom).data.astype(np.uint8), "L")

            draw = ImageDraw.Draw(img)
            draw.text((max(0, (imgWidth-70)/2), imgHeight), "{} {} {} ".format(t, round(valToSave, 2), actNum), fill=255)

            img.save('{}{:02}_{:03}.png'.format(pathOutTemp, actNum, t))

            actNum += 1

            #############################################

        actIndex += 1

    #############################################


    np.save("{}matrix_{}".format(pathOut, actSlice), tempMatrix)


    pathIn, dist, _, p, q = getBestPath(tempMatrix)

    textfile.write("{} {} {}".format(actSlice, actPos, dist))
    pathDir = "{}{}/".format(pathOutTemp, "path")

    if not needBreak:  # if cut was performed properly
        if not os.path.exists(pathDir):
            os.makedirs(pathDir)

        for i in range(len(pathIn)):
            shutil.copy2('{}{:02}_{:03}.png'.format(pathOutTemp, pathIn[i, 1], pathIn[i, 0]),
                         '{}{:03}_{:02}.png'.format(pathDir, i, pathIn[i, 1]))

    fig, ax = plt.subplots()
    cax = ax.imshow(tempMatrix, cmap=plt.cm.Reds)
    ax.set_aspect("auto")
    fig.colorbar(cax, orientation='vertical')
    plt.plot(q, p, "bo")
    plt.savefig("{}path_{}.png".format(pathOut, actSlice))

    np.savetxt("{}path_{}.txt".format(pathOut, actSlice), pathIn, fmt="%i %i") #, delimiter=',')

    actSlice += 1

#########################################
#########################################

l2normSum = np.sqrt(l2normSum)

print(l2normSum)
textfile.write("sum: {}\n".format(l2normSum))
textfile.close()
textfileInfos.close()

t1 = time.time()

print(t1-t0)
