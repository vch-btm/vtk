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
import sys
#import pdb as pdb


maskCorr = 5
pixelCorr = 2*maskCorr+5

numAverage = 13  # odd number
withCushion = not True


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def readDynpt():
    if withCushion:
        f = open(pathIn + "simulation/x.dynpt", 'rb')
    else:
        f = open(pathIn + "simulation/NC.dynpt", 'rb')


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
    pathIn = "c:/users/vch/desktop/Bredies/CASE03/"
    pathOut = "c:/users/vch/desktop/results/"
else:
    print("OS: not win")
    pathIn = "/home/horakv/Desktop/Bredies/CASE03/"
    pathOut = "/home/horakv/Desktop/results/"

textfile = open(pathOut + "list.txt", "w")
textfile.write(pathIn + '\n')

##########################################
##########################################

t0 = time.time()


###############################################################################
# Modell + Verschiebungen lesen
###############################################################################

if withCushion:
    filenames = glob.glob(pathIn + 'mesh/*.vtk')
    pathOut += "withCushionAvg/"
else:
    filenames = glob.glob(pathIn + 'mesh/*NC.vtk')
    pathOut += "noCushionAvg/"

if not os.path.exists(pathOut):
    os.makedirs(pathOut)

reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName(filenames[0])

geometryFilter = vtk.vtkGeometryFilter()
geometryFilter.SetInputConnection(reader.GetOutputPort())
geometryFilter.Update()

mPolydata = geometryFilter.GetOutput()

mMesh = vtknp.vtk_to_numpy(mPolydata.GetPoints().GetData())
mMesh /= 1000  # Daten sind in um statt in mm gegeben . Korrektur

(header, displacements) = readDynpt()

mNumberTimeSteps = len(displacements)
mPosition = getModelPosition()


#########################################
#########################################

l2normSum = 0.0

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

actSlice = 0

for actPos, actDict in dictFilesDCM.items():  # für jede Slice
    sortEntries = sorted(actDict)


    #if not actSlice == 7:
    #    actSlice += 1
    #    continue


    try:
        triggerTimeFirst = dicom.read_file(actDict[sortEntries[0]]).TriggerTime
        triggerTimeLast = dicom.read_file(actDict[sortEntries[-1]]).TriggerTime
    except AttributeError:
        triggerTimeFirst = 0.0
        triggerTimeLast = 0.0

    triggerLength = triggerTimeLast - triggerTimeFirst

    first = True
    actIndex = 0

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

            sliceFactor = pixelSpacing[2] / (numAverage - 1)

            listOfTransformFilters = []

            for slice in range(-int(numAverage / 2), int(numAverage / 2) + 1):
                matrix = vtk.vtkMatrix4x4()

                for i in range(3):
                    matrix.SetElement(i, 0, xdir[i])
                    matrix.SetElement(i, 1, ydir[i])
                    matrix.SetElement(i, 2, zdir[i])
                    matrix.SetElement(i, 3, position[i] + slice * sliceFactor * zdir[i])

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

        try:
            actTriggerTime = actDicom.TriggerTime
        except AttributeError:
            actTriggerTime = 0

        ##############################

        sumOfCuts = np.zeros(arrayDicom.shape)
        sumOfMasks = np.zeros(arrayDicom.shape, dtype=bool)

        for transformFilter in listOfTransformFilters:
            polydata = vtk.vtkPolyData()
            polydata.DeepCopy(mPolydata)

            actMesh = vtknp.vtk_to_numpy(polydata.GetPoints().GetData())

            if triggerLength == 0.0:
                pass
                # actMesh[...] = displacements[0]
            else:
                actMesh[...] = displacements[round((actTriggerTime - triggerTimeFirst) * (mNumberTimeSteps - 1) / triggerLength)]

            polydata.Modified()

            transformFilter.SetInputData(polydata)
            transformFilter.Update()

            polydata = transformFilter.GetOutput()

            polydata.GetPoints().SetData(vtknp.numpy_to_vtk(np.divide(polydata.GetPoints().GetData(), scalingVec), array_type=vtk.VTK_FLOAT))  # dividiere alle punkte durch scalingVec

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

            if pd.GetPoints() is None:  # Cut with nothing
                needBreak = True

                try:
                    actTriggerTime = actDicom.TriggerTime
                except AttributeError:
                    actTriggerTime = 0

                    textfile.write("{} {}/{}/{} {} \n".format(float("inf"), splitInfo[-4], splitInfo[-3], splitInfo[-2],
                                                              actTriggerTime))
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

            #######################################################################
            # finding cut window
            #######################################################################

    # =============================================================================
    #         yMin = int(max(0, cutBounds[0] - pixelCorr))
    #         yMax = int(min(right[0], cutBounds[1] + pixelCorr))
    #
    #         xMin = int(max(0, cutBounds[2] - pixelCorr))
    #         xMax = int(min(right[1], cutBounds[3] + pixelCorr))
    # =============================================================================

            #yMin = int(max(0, cutBounds[0] - pixelCorr))
            #yMax = int(min(right[0], cutBounds[1] + pixelCorr))

            #xMin = int(max(0, cutBounds[2] - pixelCorr))
            #xMax = int(min(right[1], cutBounds[3] + pixelCorr))

            dicomCut = arrayDicom#[xMin:xMax, yMin:yMax]

            #######################################################################
            # filling contours
            #######################################################################

            actContour = 0

            mData = np.zeros((right[0], right[1]), dtype=np.uint8)

            mSurface = cairo.ImageSurface.create_for_data(mData, cairo.FORMAT_A8, right[0], right[1])
            mContext = cairo.Context(mSurface)
            mContext.scale(1, 1)
            mContext.set_line_width(1)
            mContext.set_source_rgb(1, 1, 1)

            for c in contours:
                mContext.move_to(points[c[0], 0], points[c[0], 1])

                for p in c:
                    mContext.line_to(points[p, 0], points[p, 1])

                mContext.close_path()
                mContext.fill()

            # construct a circle for dilatation
            x = range(-maskCorr, maskCorr + 1)
            [X, Y] = np.meshgrid(x, x)
            struct = (X*X + Y*Y <= maskCorr**2)

            mask = ndimage.binary_dilation(mData, structure=struct)#[xMin:xMax, yMin:yMax]

            sumOfMasks += mask

            listOfCutData = []

            for c in contours:
                data = np.zeros((right[0], right[1]), dtype=np.uint8)

                surface = cairo.ImageSurface.create_for_data(data, cairo.FORMAT_A8, right[0], right[1])
                context = cairo.Context(surface)
                context.scale(1, 1)
                context.set_line_width(1)
                context.set_source_rgb(1, 1, 1)

                context.move_to(points[c[0], 0], points[c[0], 1])

                for p in c:
                    context.line_to(points[p, 0], points[p, 1])
                context.close_path()

                context.fill()

                tempCut = data#[xMin:xMax, yMin:yMax]

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
                    matrixP[i][j] = np.sum(listOfCutData[i] * listOfCutData[j])
                vecD[i] = np.sum(listOfCutData[i] * dicomCut)

            vecC = np.linalg.solve(matrixP, vecD)

            modelCut = ma.masked_array(np.zeros(dicomCut.shape), ~mask)

            for actContour in range(numContours):
                modelCut += listOfCutData[actContour]*vecC[actContour]

            sumOfCuts += modelCut


        #######################################################################
        # calc norm, prepare and save output
        #######################################################################

        if needBreak:
            break

        numPixels = np.sum(sumOfMasks)

        averageCut = sumOfCuts / numAverage

        a = dicomCut - averageCut
        b = a/numPixels
        c = np.sum(b)
        d = np.sqrt(c)

        l2norm = np.sqrt(np.sum(np.square(dicomCut - averageCut) / numPixels))
        l2normSum += l2norm**2

        textfile.write("{} {}/{}/{} {} {} {}\n".format(l2norm, splitInfo[-4],
                       splitInfo[-3], splitInfo[-2], actTriggerTime,
                       triggerTimeFirst, triggerLength))

        maxValArrayDicom = arrayDicom.max()

        combArray = np.concatenate((averageCut, dicomCut), axis = 1)
        imgHeight, imgWidth = combArray.shape
        img = Image.fromarray((255 * np.concatenate((combArray, np.zeros((10, imgWidth)))) / maxValArrayDicom).data.astype(np.uint8), "L")
        #pdb.set_trace()

        draw = ImageDraw.Draw(img)
        draw.text((max(0, (imgWidth-30)/2), imgHeight), str(round(l2norm, 2)), fill=255)

        img.save('{}avg{}_{}_{}_{}_{}_{}_{}_{}.png'.format(pathOut, splitInfo[-5], splitInfo[-4], splitInfo[-3], splitInfo[-2], splitInfo[-1][:-4], actTriggerTime, triggerTimeFirst, triggerTimeLast))

        #############################################

        actIndex += 1

    #############################################

    actSlice += 1

#########################################
#########################################

l2normSum = np.sqrt(l2normSum)

print(l2normSum)
textfile.write("sum: {}\n".format(l2normSum))
textfile.close()

t1 = time.time()

print(t1-t0)
