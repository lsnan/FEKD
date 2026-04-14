from models.birefnet import BiRefNet
import torch.nn as nn
from simple_unet import unet

class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.teacher_model = BiRefNet()

    def forward(self, x):
        return self.teacher_model(x)

class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.student_model = unet()

    def forward(self, x):
        return self.student_model(x)