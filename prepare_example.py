import SimpleITK as sitk
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
itk_image = sitk.ReadImage("Patient01/Patient01_AC.gipl")
ac_image = sitk.GetArrayFromImage(itk_image)
#gassuian 
sitk_gassuian = sitk.SmoothingRecursiveGaussianImageFilter()
sitk_gassuian.SetSigma(2)
sitk_gassuian.NormalizeAcrossScaleOff()
sitk_gassuian = sitk_gassuian.Execute(itk_image)
axial1 =sitk.GetArrayFromImage(sitk_gassuian)
axial1 = axial1[axial1.shape[0]//2,:,:]
plt.imshow(axial1, cmap="gray")
#median filter
sitk_median = sitk.MedianImageFilter()
sitk_median.SetRadius(5)
sitk_median = sitk_median.Execute(itk_image)
axial2 =sitk.GetArrayFromImage(sitk_median)
axial2 = axial2[axial2.shape[0]//2,:,:]
plt.imshow(axial2, cmap="gray")
#mean filter
sitk_mean = sitk.MeanImageFilter()
sitk_mean.SetRadius(5)
sitk_mean = sitk_mean.Execute(itk_image)
axial3 =sitk.GetArrayFromImage(sitk_median)
axial3 = axial3[axial3.shape[0]//2,:,:]
plt.imshow(axial3, cmap="gray")
