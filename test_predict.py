import torch
import numpy as np
from NN_hub import Lenn
from torch.autograd import Variable

model = Lenn()
model.load_state_dict(torch.load('params.pkl'))

for i in range(60):
    img = np.zeros((1, 1, 28, 28), np.float32)
    img_input = Variable(torch.from_numpy(img)).cuda()

    pre_y = model(img_input)
    pre_y = torch.max(pre_y, 1)[1].data.cpu().numpy().squeeze()

    print(pre_y)