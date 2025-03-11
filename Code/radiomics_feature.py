from __future__ import print_function
import six
import os  # needed navigate the system to get the input data
import numpy as np
import radiomics
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor  # This module is used for interaction with pyradiomics
import argparse

import csv
import cv2

def catch_features(imagePath, maskPath):
    if imagePath is None or maskPath is None:  # Something went wrong, in this case PyRadiomics will also log an error
        raise Exception('Error getting testcase!')  # Raise exception to prevent cells below from running in case of "run all"
    settings = {}
    settings['binWidth'] = 25
    settings['resampledPixelSpacing'] = None
    settings['interpolator'] = 'sitkBSpline'
    settings['verbose'] = True
    settings['geometryTolerance'] = 1e-5

    
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    
    #extractor = featureextractor.RadiomicsFeatureExtractor()
    #print('Extraction parameters:\n\t', extractor.settings)

    #extractor.enableImageTypeByName('LoG')
    #extractor.enableImageTypeByName('Wavelet')

    # Disable all classes
    extractor.disableAllFeatures()

    # First Order Feature
    extractor.enableFeaturesByName(firstorder=['Mean', 'StandardDeviation',  'Skewness', 'Kurtosis', 'Entropy', 'Minimum', 'Maximum', 'Median', 'InterquartileRange', 'Range'])
    
    # Shape Feature
    #extractor.enableFeaturesByName(shape2D=['MeshSurface', 'PixelSurface', 'Perimeter', 'PerimeterSurfaceRatio', 'Sphericity','SphericalDisproportion','MaximumDiameter',
    #                                        'MajorAxisLength', 'MinorAxisLength', 'Elongation'])
    
    # GLCM Feature
    #extractor.enableFeaturesByName(glcm=['JointEnergy', 'Contrast',  'Correlation', 'SumSquares', 'SumAverage', 'JointEntropy', 'DifferenceVariance', 'DifferenceEntropy', 'DifferenceAverage',
    #                                     'Imc1', 'Imc2', 'Autocorrelation','ClusterShade', 'ClusterProminence', 'MaximumProbability', 'InverseVariance'])
    

    feature_cur = []
    feature_name = []
    itkimage = sitk.ReadImage(imagePath, sitk.sitkUInt8)
    itklabel = sitk.ReadImage(maskPath, sitk.sitkUInt8)
    result = extractor.execute(itkimage, itklabel)
 
    #result = extractor.execute(imagePath, maskPath)
    for key, value in six.iteritems(result):

        feature_name.append(key)
        feature_cur.append(value)

    name = feature_name
    name = np.array(name)
    
    return feature_cur[-10:], name[-10:]


file_path = pd.read_csv('../images/SYSUCC/img_path.csv')
label_path_all = np.array(file_path['label_path'])
images_path_all = np.array(file_path['img_path'])

try:
    feature, name = catch_features(images_path_all, label_path_all)
    #print(feature, name)
    with open("../Data/radiomics_features/first_sysucc.csv", 'a', newline='') as file:  
        writer = csv.writer(file)
        writer.writerow(name)   
        writer.writerow([images_path_all, feature])

except Exception as e:
    print(e)    
