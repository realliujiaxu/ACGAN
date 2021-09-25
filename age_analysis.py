from deepface import DeepFace
import numpy as np
from tqdm import tqdm

ages_list = []
for i in tqdm(range(40)):
    ages = []
    for j in range(11):
        obj = DeepFace.analyze(img_path = "results/face256random/age_moving/{}-{}..jpg".format(
            str(i).zfill(4), str(j).zfill(2)), actions = ['age'])
        ages.append(obj["age"])
    ages_list.append(ages)

ages_numpy = np.array(ages_list)
np.save('age_moving.npy', ages_numpy)
age_mean = ages_numpy.mean(axis=0)
print(age_mean)

import matplotlib.pyplot as plt
x1 = np.linspace(0, 1, 11)
plt.plot(x1,age_mean,'g+-')
plt.title('Age evaluation')
plt.xlabel('Prediction')
plt.ylabel('Attribute')
plt.legend()
plt.savefig('./age_moving.png')
plt.show()
