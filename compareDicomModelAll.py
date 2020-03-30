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
#import pdb as pdb


maskCorr = 5
pixelCorr = 2*maskCorr+5


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
    pathIn = "c:/users/vch/desktop/Bredies/CASE01/"
    pathOut = "c:/users/vch/desktop/results/"


else:
    print("OS: not win")
    pathIn = "/home/horakv/Desktop/Bredies/CASE01/"
    pathOut = "/home/horakv/Desktop/results/"

if not os.path.exists(pathOut):
    os.makedirs(pathOut)


textfile = open(pathOut + "list.txt", "w")
textfile.write(pathIn + '\n')

##########################################
##########################################

t0 = time.time()


###############################################################################
# Modell + Verschiebungen lesen
###############################################################################

filenames = glob.glob(pathIn + 'mesh/*.vtk')
#filenames = glob.glob(pathIn + 'mesh/*cutted.vtk')

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
            posOri = str(actDicom.ImagePositionPatient + actDicom.ImageOrientationPatient)

            if (posOri not in dictFilesDCM):
                dictFilesDCM[posOri] = {}
            dictFilesDCM[posOri][actDicom.InstanceNumber] = os.path.join(dirName, filename)
#########################################

actSlice = 0

textfileInfos = open(pathOut + "listInfo.txt", "w")

for actPos, actDict in dictFilesDCM.items():  # für jede Slice
    sortEntries = sorted(actDict)

    textfileInfos.write("{} {}\n".format(pathIn, actPos))

    pathOutTemp = "{}{}/".format(pathOut, actSlice)

    if not os.path.exists(pathOutTemp):
        os.makedirs(pathOutTemp)


    if actSlice != 0:
        actSlice += 1
        continue

    first = True
    actIndex = 0

    seriesOfDicoms = []


    transformFilter = 0



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

    tempMatrix = np.zeros((mNumberTimeSteps, len(sortEntries)))

    # textfileTemp.write("{} {}\n".format(mNumberTimeSteps, len(sortEntries)))


    for t in range(mNumberTimeSteps):
        print(actSlice, t)

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

        if pd.GetPoints() is None:
            textfile.write(
                "{} {}/{}/{} {} {}\n".format(float("inf"), splitInfo[-4], splitInfo[-3], splitInfo[-2],
                                                actTriggerTime, t))
            continue

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

            origContours = contours

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

        yMin = int(max(0, cutBounds[0] - pixelCorr))
        yMax = int(min(right[0], cutBounds[1] + pixelCorr))

        xMin = int(max(0, cutBounds[2] - pixelCorr))
        xMax = int(min(right[1], cutBounds[3] + pixelCorr))


        actNum = 0

        for actDicom, arrayDicom in seriesOfDicoms:  # für jedes einzelne Bild
            dicomCut = arrayDicom[xMin:xMax, yMin:yMax]


            try:
                actTriggerTime = actDicom.TriggerTime
            except AttributeError:
                actTriggerTime = 0

            ##############################


            #######################################################################
            # filling contours
            #######################################################################

            contour = origContours.copy()

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

            mask = ndimage.binary_dilation(mData, structure=struct)[xMin:xMax, yMin:yMax]

            numPixels = np.sum(mask)

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

            textfile.write("{} {}/{}/{} {} {}\n".format(l2norm, splitInfo[-4],
                           splitInfo[-3], splitInfo[-2], actTriggerTime, t))


            tempMatrix[t][actNum] = l2norm
            #textfileTemp.write("{} {} {}\n".format(t, actNum, l2norm))

            maxValArrayDicom = arrayDicom.max()

            combArray = np.concatenate((modelCut, dicomCut), axis = 1)
            imgHeight, imgWidth = combArray.shape
            img = Image.fromarray((255 * np.concatenate((combArray, np.zeros((10, imgWidth)))) / maxValArrayDicom).data.astype(np.uint8), "L")
            #pdb.set_trace()

            draw = ImageDraw.Draw(img)
            draw.text((max(0, (imgWidth-30)/2), imgHeight), str(round(l2norm, 2)), fill=255)

            img.save('{}{}_{}_{}_{}_{}_{}_{}.png'.format(pathOutTemp, splitInfo[-5], splitInfo[-4], splitInfo[-3], splitInfo[-2], splitInfo[-1][:-4], actTriggerTime, t))

            actNum += 1

            #############################################

        actIndex += 1

    #############################################

    #textfileTemp.close()
    np.save("matrixF_{}".format(actPos), tempMatrix)

    actSlice += 1

#########################################
#########################################

l2normSum = np.sqrt(l2normSum)

print(l2normSum)
textfile.write("sum: {}\n".format(l2normSum))
textfile.close()

t1 = time.time()

print(t1-t0)
