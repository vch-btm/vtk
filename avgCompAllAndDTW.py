# coding=utf-8

import glob
import vtk
import vtk.util.numpy_support as vtknp
import platform
import os
import sys
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

txtCase = "01"

maskCorr = 5
pixelCorr = 2 * maskCorr + 5

numAverage = 1  # odd number
withCushion = not True

print(numAverage)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def dtw(act_matrix):
    global pathIn

    r, o = act_matrix.shape
    D0 = np.zeros((r + 1, o + 1))
    D0[0, 1:] = np.inf
    D0[1:, 0] = np.inf
    D1 = D0[1:, 1:]  # view
    for m in range(r):
        for n in range(o):
            D1[m, n] = act_matrix[m, n]
    C = D1.copy()
    for m in range(r):
        for n in range(o):
            D1[m, n] += min(D0[m, n], D0[m, n + 1], D0[m + 1, n])

    p, q, bestPath = _traceback(D0)

    return D1[-1, -1] / sum(D1.shape), C, D1, p, q, bestPath


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def _traceback(matrix_d):
    global pathIn

    m, n = np.array(matrix_d.shape) - 2

    p, q = [m], [n]
    bestPath = []
    bestPath.insert(0, (m, n))

    while (m > 0) or (n > 0):
        tb = np.argmin((matrix_d[m, n], matrix_d[m, n + 1], matrix_d[m + 1, n]))
        if tb == 0:
            m -= 1
            n -= 1
        elif tb == 1:
            m -= 1
        else:  # (tb == 2):
            n -= 1

        bestPath.insert(0, (m, n))
        p.insert(0, m)
        q.insert(0, n)

    return np.array(p), np.array(q), np.array(bestPath)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def getBestPath(act_matrix):
    bestI = -1
    bestMat1 = 0
    # bestMat2 = 0
    bestp = 0
    bestq = 0
    best_dist = np.inf
    best_path = 0

    length = len(act_matrix[0])

    for iTemp in range(1, length + 1):
        rolledMatrix = np.roll(act_matrix, iTemp, axis=1)

        dist, mat1, mat2, p, q, path = dtw(rolledMatrix)

        if dist < best_dist:
            bestI = iTemp
            bestMat1 = mat1
            # bestMat2 = mat2
            bestp = p
            bestq = q
            best_dist = dist
            best_path = path

    for iTemp in range(len(best_path)):
        best_path[iTemp, 1] = (best_path[iTemp, 1] - bestI + length) % length
        bestq[iTemp] = (bestq[iTemp] - bestI + length) % length

    return best_path, best_dist, bestMat1, bestp, bestq


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def readDynpt():
    f = open(pathIn + "simulation/x.dynpt", 'rb')
    act_header = dict(re.findall(r"(\w*):(\w*)", f.read(1024).decode('utf-8')))

    shapeTest = [int(act_header['t']), int(act_header['x']), 3]

    act_data = np.fromfile(f, dtype=np.float32)

    if act_header['unites_x'] == "um":
        act_data /= 1000

    return act_header, act_data.reshape(shapeTest)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def getModelPosition():
    global fileList

    minZPos = float("inf")

    fileList = glob.glob(pathIn + 'segmentation/*/*/*.dcm')

    tempDicom = 0

    for act_filename in fileList:
        tempDicom = dicom.read_file(act_filename)
        actZPos = tempDicom.ImagePositionPatient[2]

        if actZPos < minZPos:
            minZPos = actZPos

    return [tempDicom.ImagePositionPatient[0],
            tempDicom.ImagePositionPatient[1],
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

if withCushion:
    filenames = glob.glob(pathIn + 'mesh/*.vtk')
    pathOut += "withCushion/avg_{}/".format(numAverage)
else:
    filenames = glob.glob(pathIn + 'simulation/vtkNC/*.vtk')
    pathOut += "noCushion/avg_{}/".format(numAverage)

if not os.path.exists(pathOut):
    os.makedirs(pathOut)

textfile = open(pathOut + "list.txt", "w")
textfile.write("l2norm sliceNr triggerTimeDicom tModel\n")

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

            if posOri not in dictFilesDCM:
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
    if not actSlice > 10:
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

        if first:  # organisiere Metadaten + arrayDicom anlegen
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
                sys.exit("no slice thickness given")

            position = actDicom.ImagePositionPatient
            orientation = actDicom.ImageOrientationPatient

            xdir = orientation[0:3]
            ydir = orientation[3:6]
            zdir = [0.0, 0.0, 0.0]

            vtk.vtkMath.Cross(xdir, ydir, zdir)
            zdir = normalize(zdir)

            if numAverage == 1:
                sliceFactor = 1
            else:
                sliceFactor = pixelSpacing[2] / (numAverage - 1)

            listOfTransformFilters = []

            for actSliceForAvg in range(-int(numAverage / 2), int(numAverage / 2) + 1):
                matrix = vtk.vtkMatrix4x4()

                for i in range(3):
                    matrix.SetElement(i, 0, xdir[i])
                    matrix.SetElement(i, 1, ydir[i])
                    matrix.SetElement(i, 2, zdir[i])
                    matrix.SetElement(i, 3, position[i] + actSliceForAvg * sliceFactor * zdir[i])

                scalingVec = [pixelSpacing[0], pixelSpacing[1], (pixelSpacing[0] + pixelSpacing[1]) / 2]

                matrix.Invert()

                translation = vtk.vtkTransform()
                translation.SetMatrix(matrix)
                translation.Translate(mPosition)

                transformFilter = vtk.vtkTransformPolyDataFilter()
                transformFilter.SetTransform(translation)

                listOfTransformFilters.append(transformFilter)

        arrayDicom = np.zeros((actDicom.Rows, actDicom.Columns), dtype=float)

        arrayDicom[:, :] = actDicom.pixel_array

        arrayDicom = resSlope * arrayDicom + resIntercept

        seriesOfDicoms.append((actDicom, arrayDicom))

    allContours = []
    allPoints = []

    yMin = np.inf
    yMax = -np.inf

    xMin = np.inf
    xMax = -np.inf

    for t in range(mNumberTimeSteps):
        someContours = []
        somePoints = []

        for transformFilter in listOfTransformFilters:
            polydata = vtk.vtkPolyData()
            polydata.DeepCopy(mPolydata)

            actMesh = vtknp.vtk_to_numpy(polydata.GetPoints().GetData())
            actMesh[...] = displacements[t]

            polydata.Modified()

            transformFilter.SetInputData(polydata)
            transformFilter.Update()

            polydata = transformFilter.GetOutput()

            polydata.GetPoints().SetData(vtknp.numpy_to_vtk(np.divide(polydata.GetPoints().GetData(), scalingVec),
                                                            array_type=vtk.VTK_FLOAT))
            # dividiere alle punkte durch scalingVec

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

            needBreak = False

            if pd.GetPoints() is None:
                needBreak = True

                try:
                    actTriggerTime = actDicom.TriggerTime
                except AttributeError:
                    actTriggerTime = 0

                textfile.write("{} {} {} {}\n".format(float("inf"), actSlice, actTriggerTime, t))

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
            neighbors = {bestP: len(bestQ) for (bestP, bestQ) in pp.items()}
            starters = [bestP for (bestP, bestQ) in neighbors.items() if np.mod(bestQ, 2) == 1]
            # traverse possible pathIns
            contours = []
            while len(pp) > 0:
                curcontour = []
                if len(starters) > 0:
                    curpoint = starters.pop()
                else:
                    curpoint = list(pp.keys())[0]
                while curpoint in pp:
                    curcontour.append(curpoint)
                    nextpoint = pp[curpoint].pop()
                    if len(pp[curpoint]) == 0:
                        pp.pop(curpoint)
                    pp[nextpoint].remove(curpoint)
                    if len(pp[nextpoint]) == 0:
                        pp.pop(nextpoint)
                    curpoint = nextpoint
                if len(curcontour) > 0:
                    contours.append(curcontour)

            yMin = int(min(int(max(0, cutBounds[0] - pixelCorr)), yMin))
            yMax = int(max(int(min(right[0], cutBounds[1] + pixelCorr)), yMax))

            xMin = int(min(int(max(0, cutBounds[2] - pixelCorr)), xMin))
            xMax = int(max(int(min(right[1], cutBounds[3] + pixelCorr)), xMax))

            someContours.append(contours)
            somePoints.append(points)

        allContours.append(someContours)
        allPoints.append(somePoints)

    for t in range(mNumberTimeSteps):
        actNum = 0

        if needBreak:
            break

        for actDicom, arrayDicom in seriesOfDicoms:  # für jedes einzelne Bild
            dicomCut = arrayDicom[xMin:xMax, yMin:yMax]

            sumOfCuts = np.zeros(dicomCut.shape)
            sumOfMasks = np.zeros(dicomCut.shape, dtype=bool)

            for avCount in range(numAverage):
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

                for c in allContours[t][avCount]:
                    mContext.move_to(allPoints[t][avCount][c[0], 0], allPoints[t][avCount][c[0], 1])

                    for bestP in c:
                        mContext.line_to(allPoints[t][avCount][bestP, 0], allPoints[t][avCount][bestP, 1])

                    mContext.close_path()
                    mContext.fill()

                mask = ndimage.binary_dilation(mData, structure=struct)[xMin:xMax, yMin:yMax]

                sumOfMasks += mask

                listOfCutData = []

                for c in allContours[t][avCount]:
                    data = np.zeros((right[0], right[1]), dtype=np.uint8)

                    surface = cairo.ImageSurface.create_for_data(data, cairo.FORMAT_A8, right[0], right[1])
                    context = cairo.Context(surface)
                    context.scale(1, 1)
                    context.set_line_width(1)
                    context.set_source_rgb(1, 1, 1)

                    context.move_to(allPoints[t][avCount][c[0], 0], allPoints[t][avCount][c[0], 1])

                    for bestP in c:
                        context.line_to(allPoints[t][avCount][bestP, 0], allPoints[t][avCount][bestP, 1])
                    context.close_path()

                    context.fill()

                    tempCut = data[xMin:xMax, yMin:yMax]

                    if not np.sum(tempCut) == 0.0:
                        listOfCutData.append(tempCut / 255)

                listOfCutData.append(np.ones(dicomCut.shape))  # weißes Grundbild

                numContours = len(listOfCutData)

                dicomCut = ma.masked_array(dicomCut, ~mask)

                for i in range(numContours):
                    listOfCutData[i] = ma.masked_array(listOfCutData[i], ~mask)

                matrixP = np.zeros((numContours, numContours))
                vecD = np.zeros(numContours)

                for i in range(numContours):
                    for j in range(numContours):
                        matrixP[i][j] = np.sum(listOfCutData[i] * listOfCutData[j])
                    vecD[i] = np.sum(listOfCutData[i] * dicomCut)

                vecC = np.linalg.solve(matrixP, vecD)

                modelCut = ma.masked_array(np.zeros(dicomCut.shape), ~mask)

                for actContour in range(numContours):
                    modelCut += listOfCutData[actContour] * vecC[actContour]

                sumOfCuts += modelCut

            #######################################################################
            # calc norm, prepare and save output
            #######################################################################

            if needBreak:
                break

            numPixels = np.sum(sumOfMasks)

            averageCut = sumOfCuts / numAverage

            l2norm = np.sqrt(np.sum(np.square(dicomCut - averageCut) / numPixels))
            l2normSum += l2norm ** 2

            # valToSave = 0

            # if withSquare:
            #    valToSave = l2norm**2
            # else:
            valToSave = l2norm

            textfile.write("{} {} {} {}\n".format(l2norm, actSlice, actTriggerTime, t))

            tempMatrix[t][actNum] = valToSave

            maxValArrayDicom = arrayDicom.max()

            combArray = np.concatenate((averageCut, dicomCut), axis=1)
            imgHeight, imgWidth = combArray.shape
            img = Image.fromarray(
                (255 * np.concatenate((combArray, np.zeros((10, imgWidth)))) / maxValArrayDicom).data.astype(np.uint8),
                "L")

            draw = ImageDraw.Draw(img)
            draw.text((max(0, (imgWidth - 70) / 2), imgHeight), "{} {} {} ".format(t, round(valToSave, 2), actNum),
                      fill=255)

            img.save('{}{:02}_{:03}.png'.format(pathOutTemp, actNum, t))

            actNum += 1

            #############################################

        actIndex += 1

    #############################################

    np.save("{}matrix_{}".format(pathOut, actSlice), tempMatrix)

    pathIn, bestDist, _, bestP, bestQ = getBestPath(tempMatrix)

    textfile.write("{} {} {}".format(actSlice, actPos, bestDist))
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
    ax.set_ylabel('time steps of the model')
    ax.yaxis.set_label_coords(-.1, .5)
    ax.set_xlabel('time steps of the dicom image')
    ax.xaxis.set_label_coords(.5, -.1)
    fig.colorbar(cax, orientation='vertical')
    plt.plot(bestQ, bestP, "bo")
    plt.savefig("{}path_{}.png".format(pathOut, actSlice))

    np.savetxt("{}path_{}.txt".format(pathOut, actSlice), pathIn, fmt="%i %i")  # , delimiter=',')

    actSlice += 1

#########################################
#########################################

l2normSum = np.sqrt(l2normSum)

print(l2normSum)
textfile.write("sum: {}\n".format(l2normSum))
textfile.close()
textfileInfos.close()

t1 = time.time()

print(t1 - t0)
