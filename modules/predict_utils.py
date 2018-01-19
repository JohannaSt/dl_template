from modules import io
from modules import layers as tf_util
from modules import vascular_data as sv
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
import argparse
import scipy
import importlib
import pandas as pd
import pickle
from tqdm import tqdm
from vtk import vtkImageExport
from vtk.util import numpy_support
import vtk

def VTKPDPointstoNumpy(pd):
	'''
	function to convert the points data of a vtk polydata object to a numpy array

	args:
		@a pd: vtk.vtkPolyData object
	'''
	return numpy_support.vtk_to_numpy(pd.GetPoints().GetData())

def VTKNumpytoSP(img_):
    img = img_.T

    H,W = img.shape

    sp = vtk.vtkStructuredPoints()
    sp.SetDimensions(H,W,1)
    sp.AllocateScalars(10,1)
    for i in range(H):
        for j in range(W):
            v = img[i,j]
            sp.SetScalarComponentFromFloat(i,j,0,0,v)

    return sp

def marchingSquares(img, iso=0.0, mode='center'):
    s = img.shape
    alg = vtk.vtkMarchingSquares()

    sp = VTKNumpytoSP(img)

    alg.SetInputData(sp)
    alg.SetValue(0,iso)
    alg.Update()
    pds = alg.GetOutput()

    a = vtk.vtkPolyDataConnectivityFilter()
    a.SetInputData(pds)

    if mode=='center':
        a.SetExtractionModeToClosestPointRegion()
        a.SetClosestPoint(float(s[0])/2,float(s[1])/2,0.0)

    elif mode=='all':
        a.SetExtractionModeToAllRegions()

    a.Update()
    pds = a.GetOutput()

    if pds.GetPoints() is None:
        return np.asarray([[0.0,0.0],[0.0,0.0]])
    else:
        pds = VTKPDPointstoNumpy(pds)
        if len(pds) <= 1:
            return np.asarray([[0.0,0.0],[0.0,0.0]])
        return pds
