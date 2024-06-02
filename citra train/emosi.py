import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


import os
for dirname, _, filenames in os.walk('C:\LIngga\CITRA DIGITAL\images'):
    for filename in filenames:
        print(os.path.join(dirname, filename))