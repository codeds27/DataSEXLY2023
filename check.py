import numpy as np
import os

org_dir = r'F:\test\PythonProject\XLYTF\data'
predict_dir = r'F:\test\PythonProject\XLYTF\best_model\WaveNet'

# [3549,12,170,1]
x = np.load(os.path.join(org_dir, 'test_x.npy'))
print(np.mean(x))
print(np.std(x))

predict = np.load(os.path.join(predict_dir, 'predict.npy'))
print(np.mean(predict))
print(np.std(predict))