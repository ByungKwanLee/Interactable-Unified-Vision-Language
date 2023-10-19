# %%

import torch
import random
import numpy as np

# 랜덤값 모두 고정
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# 데이터셋을 준비하기 위해서 필요한 라이브러리 (지금 당장 자세하게 알 필요 없음, 자연스럽게 숙지 됨)
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

# 다운로드 경로 정의
download_root = './MNIST_DATASET'

# MNIST 데이터셋을 학습 데이터셋과 테스트 데이터셋으로 나누기
train_dataset = MNIST(download_root, transform=transforms.ToTensor(), train=True, download=True)
test_dataset = MNIST(download_root, transform=transforms.ToTensor(), train=False, download=True)

# 데이터 셋 준비 (각종 기능들 있음, 데이터 셔플)
train_loader = DataLoader(dataset=train_dataset, 
                         batch_size=128, # 128개씩 입력값에 들어간다 (계산의 효율성, 최적화 효율성)
                         shuffle=True) 

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=128, # 128개씩 입력값에 들어간다 (계산의 효율성, 최적화 효율성)
                         shuffle=True)



""" 구분선 """


import torch.nn as nn # neural network (nn) 패키지 불러오기
import torch.nn.functional as F # 각종 함수들 불러오기

# nn.Module 을 상속해서 nn.Module 안에 있는 각종 기능 모두 사용가능 (학습할 수 있는 툴이 있음)
class Model(nn.Module):
    # 모델 초기화하는 함수 __init__ 의 역할 동일
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 512) # 784 (28x28이 MNIST 이미지 사이즈)개의 노드에서 512개의 노드로 신경망 생성
        self.linear2 = nn.Linear(512, 256) # 512개의 노드에서 256개의 노드로 신경망 생성
        self.linear3 = nn.Linear(256, 10) # 256개의 노드에서 10개의 노드로 신경망 생성

    # 입력값을 인공 신경망에 흘려보내는 함수
    def forward(self, x):
        x = self.linear1(x) # 784 노드에서 512 노드로 가는 레이어
        x = F.relu(x) # 활성화 함수 (비선형성 성질 만들기)
        x = self.linear2(x) # 512 노드에서 256 노드로 가는 레이어
        x = F.relu(x) # 활성화 함수 (비선형성 성질 만들기)
        x = self.linear3(x) # 256 노드에서 10 노드로 가는 레이어
        return x


""" 구분선 """
import matplotlib.pyplot as plt
img, label = next(iter(train_loader))
print(img.shape)
print(label.shape)

# 1번째 이미지
plt.imshow(img[0].permute(1,2,0), cmap='grey')
plt.title(f'Hand-Written Image {label[0]}')
plt.show()

# 2번째 이미지
plt.imshow(img[1].permute(1,2,0), cmap='grey')
plt.title(f'Hand-Written Image {label[1]}')
plt.show()

# 3번째 이미지
plt.imshow(img[2].permute(1,2,0), cmap='grey')
plt.title(f'Hand-Written Image {label[2]}')
plt.show()

# 4번째 이미지
plt.imshow(img[3].permute(1,2,0), cmap='grey')
plt.title(f'Hand-Written Image {label[3]}')
plt.show()



""" 구분선 """

model=Model() # 인공신경망 생성
y=model(img.view(128, -1)) # 인공신경망에 이미지 넣어 출력값 생성하기
print(y.shape) # [128, 10]


""" 구분선 """
import torch # PyTorch 패키지 불러오기
# 최적화 기법 설정, 학습률과 파라미터 정규화 설정
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.001)
# 학습률 스케쥴링 설정
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5) 

# 학습횟수
epochs = 2

# 모델을 학습할 수 있는 상태로 만들기
model.train()

# 학습 진행 속도 보는 패키지 불러오기
from tqdm import tqdm
# 각종 함수들 불러오기
import torch.nn.functional as F 

# 딥러닝 모델 학습하기
for epoch in range(epochs):

    print(f"현재 Epoch: {epoch+1}") # 현재 학습횟수 출력하기

    correct = 0
    total = 0

    for img, label in tqdm(train_loader):

        # 입력 이미지 신경망 통과
        y=model(img.view(img.shape[0], -1))

        # 최적화
        optimizer.zero_grad()
        loss = F.cross_entropy(y, label)
        loss.backward()
        optimizer.step()

        # 정확도 평가 기본 작업
        correct += (y.argmax(dim=1) == label).sum()
        total += label.numel()

    # 학습 데이터셋에 대한 정확도 평가하기
    acc = correct / total * 100
    print(f"현재 Epoch {epoch+1}의 정확도는 {acc:.2f} 입니다.\n")

    # 스케쥴러 스텝
    scheduler.step()


""" 구분선 """

# 테스트 셋의 정확도 구하기
correct = 0
total = 0
model.eval()
for img, label in tqdm(test_loader):

    # 입력 이미지 신경망 통과
    y=model(img.view(img.shape[0], -1))

    # 정확도 평가 기본 작업
    correct += (y.argmax(dim=1) == label).sum()
    total += label.numel()

# 학습 데이터셋에 대한 정확도 평가하기
acc = correct / total * 100
print(f"테스트 셋의 정확도는 {acc:.2f} 입니다.")

torch.save(model.state_dict(), "model.pth")

""" 구분선 """
# %%
import torch.nn as nn # neural network (nn) 패키지 불러오기
import torch.nn.functional as F # 각종 함수들 불러오기

# nn.Module 을 상속해서 nn.Module 안에 있는 각종 기능 모두 사용가능 (학습할 수 있는 툴이 있음)
class Model(nn.Module):
    # 모델 초기화하는 함수 __init__ 의 역할 동일
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 512) # 784 (28x28이 MNIST 이미지 사이즈)개의 노드에서 512개의 노드로 신경망 생성
        self.linear2 = nn.Linear(512, 256) # 512개의 노드에서 256개의 노드로 신경망 생성
        self.linear3 = nn.Linear(256, 10) # 256개의 노드에서 10개의 노드로 신경망 생성

    # 입력값을 인공 신경망에 흘려보내는 함수
    def forward(self, x):
        x = self.linear1(x) # 784 노드에서 512 노드로 가는 레이어
        x = F.relu(x) # 활성화 함수 (비선형성 성질 만들기)
        x = self.linear2(x) # 512 노드에서 256 노드로 가는 레이어
        x = F.relu(x) # 활성화 함수 (비선형성 성질 만들기)
        x = self.linear3(x) # 256 노드에서 10 노드로 가는 레이어
        return x

model=Model() # 인공신경망 생성
msg = model.load_state_dict(torch.load("model.pth"))
print(msg)


# 개인 사진 업로드
from PIL import Image
img = 255-np.array(Image.open("2.png"))[...,:3].mean(axis=2)
img = torch.from_numpy(img).float()

plt.imshow(img, cmap='grey')
plt.title(f'Hand-Written Image')
plt.show()

# 이미지 해상도 변경 (28 x 28)
img = F.interpolate(img.unsqueeze(0).unsqueeze(1), size=(28, 28))

# 모델의 예측 결과 
model.eval()
y = model(img.view(1, -1))
print(f"모델의 예측 숫자는 {y.argmax(dim=1)} 입니다.")


# %%
import torch
import random
import numpy as np
import torch.nn as nn # neural network (nn) 패키지 불러오기
import torch.nn.functional as F # 각종 함수들 불러오기


# 랜덤값 모두 고정
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


# nn.Module 을 상속해서 nn.Module 안에 있는 각종 기능 모두 사용가능 (학습할 수 있는 툴이 있음)
class Model(nn.Module):
    # 모델 초기화하는 함수 __init__ 의 역할 동일
    def __init__(self):
        super().__init__()
        """
        이 부분 수정하세요
        """
        self.linear1 = nn.Linear(784, 256) # 784 (28x28이 MNIST 이미지 사이즈)개의 노드에서 512개의 노드로 신경망 생성
        self.linear2 = nn.Linear(256, 256) # 512개의 노드에서 256개의 노드로 신경망 생성
        self.linear3 = nn.Linear(256, 784) # 256개의 노드에서 10개의 노드로 신경망 생성
        self.linear4 = nn.Linear(784, 128) # 256개의 노드에서 10개의 노드로 신경망 생성
        self.linear5 = nn.Linear(128, 10) # 256개의 노드에서 10개의 노드로 신경망 생성
        self.linear6 = nn.Linear(10, 10) # 256개의 노드에서 10개의 노드로 신경망 생성

    # 입력값을 인공 신경망에 흘려보내는 함수
    def forward(self, x):
        """
        이 부분 수정하세요
        """
        x = self.linear1(x) # 784 노드에서 512 노드로 가는 레이어
        x = F.relu(x) # 활성화 함수 (비선형성 성질 만들기)
        x = self.linear2(x) # 512 노드에서 256 노드로 가는 레이어
        x = F.relu(x) # 활성화 함수 (비선형성 성질 만들기)
        x = self.linear3(x) # 256 노드에서 10 노드로 가는 레이어
        x = F.relu(x) # 활성화 함수 (비선형성 성질 만들기)
        x = self.linear4(x) # 256 노드에서 10 노드로 가는 레이어
        x = F.relu(x) # 활성화 함수 (비선형성 성질 만들기)
        x = self.linear5(x) # 256 노드에서 10 노드로 가는 레이어
        x = F.relu(x) # 활성화 함수 (비선형성 성질 만들기)
        x = self.linear6(x) # 256 노드에서 10 노드로 가는 레이어
        return x


"""
아래 코드 변경하지 마세요.
"""

# 데이터셋을 준비하기 위해서 필요한 라이브러리 (지금 당장 자세하게 알 필요 없음, 자연스럽게 숙지 됨)
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# 다운로드 경로 정의
download_root = './MNIST_DATASET'

# MNIST 데이터셋을 학습 데이터셋과 테스트 데이터셋으로 나누기
train_dataset = MNIST(download_root, transform=transforms.ToTensor(), train=True, download=True)
test_dataset = MNIST(download_root, transform=transforms.ToTensor(), train=False, download=True)

# 데이터 셋 준비 (각종 기능들 있음, 데이터 셔플)
train_loader = DataLoader(dataset=train_dataset, 
                         batch_size=128, # 128개씩 입력값에 들어간다 (계산의 효율성, 최적화 효율성)
                         shuffle=True) 

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=128, # 128개씩 입력값에 들어간다 (계산의 효율성, 최적화 효율성)
                         shuffle=True)

import matplotlib.pyplot as plt
img, label = next(iter(test_loader))
model=Model() # 인공신경망 생성
y=model(img.view(128, -1)) # 인공신경망에 이미지 넣어 출력값 생성하기
print(y[125]) # [128, 10]


# %%
import torch
import random
import numpy as np

# 랜덤값 모두 고정
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# 데이터셋을 준비하기 위해서 필요한 라이브러리 (지금 당장 자세하게 알 필요 없음, 자연스럽게 숙지 됨)
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

# 다운로드 경로 정의
download_root = './MNIST_DATASET'

# MNIST 데이터셋을 학습 데이터셋과 테스트 데이터셋으로 나누기
train_dataset = MNIST(download_root, transform=transforms.ToTensor(), train=True, download=True)
test_dataset = MNIST(download_root, transform=transforms.ToTensor(), train=False, download=True)

# 데이터 셋 준비 (각종 기능들 있음, 데이터 셔플)
train_loader = DataLoader(dataset=train_dataset, 
                         batch_size=128, # 128개씩 입력값에 들어간다 (계산의 효율성, 최적화 효율성)
                         shuffle=True) 

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=128, # 128개씩 입력값에 들어간다 (계산의 효율성, 최적화 효율성)
                         shuffle=True)


import torch.nn as nn # neural network (nn) 패키지 불러오기
import torch.nn.functional as F # 각종 함수들 불러오기

# nn.Module 을 상속해서 nn.Module 안에 있는 각종 기능 모두 사용가능 (학습할 수 있는 툴이 있음)
class Model(nn.Module):
    # 모델 초기화하는 함수 __init__ 의 역할 동일
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 512) # 784 (28x28이 MNIST 이미지 사이즈)개의 노드에서 512개의 노드로 신경망 생성
        self.linear2 = nn.Linear(512, 256) # 512개의 노드에서 256개의 노드로 신경망 생성
        self.linear3 = nn.Linear(256, 10) # 256개의 노드에서 10개의 노드로 신경망 생성

    # 입력값을 인공 신경망에 흘려보내는 함수
    def forward(self, x):
        x = self.linear1(x) # 784 노드에서 512 노드로 가는 레이어
        x = F.relu(x) # 활성화 함수 (비선형성 성질 만들기)
        x = self.linear2(x) # 512 노드에서 256 노드로 가는 레이어
        x = F.relu(x) # 활성화 함수 (비선형성 성질 만들기)
        x = self.linear3(x) # 256 노드에서 10 노드로 가는 레이어
        return x


model = Model() # 모델 생성

# 최적화 기법 설정, 학습률과 파라미터 정규화 설정
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.001)
# 학습률 스케쥴링 설정
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5) 

# 학습횟수
epochs = 2

# 모델을 학습할 수 있는 상태로 만들기
model.train()

# 학습 진행 속도 보는 패키지 불러오기
from tqdm import tqdm
# 각종 함수들 불러오기
import torch.nn.functional as F 

# 딥러닝 모델 학습하기
for epoch in range(epochs):

    print(f"현재 Epoch: {epoch+1}") # 현재 학습횟수 출력하기

    correct = 0
    total = 0

    for img, label in tqdm(train_loader):

        # 입력 이미지 신경망 통과
        y=model(img.view(img.shape[0], -1))

        # 최적화
        optimizer.zero_grad()
        loss = F.cross_entropy(y, label)
        loss.backward()
        optimizer.step()

        # 정확도 평가 기본 작업
        correct += (y.argmax(dim=1) == label).sum()
        total += label.numel()

    # 학습 데이터셋에 대한 정확도 평가하기
    acc = correct / total * 100
    print(f"현재 Epoch {epoch+1}의 정확도는 {acc:.2f} 입니다.\n")

    # 스케쥴러 스텝
    scheduler.step()

# 테스트 셋의 정확도 구하기
correct = 0
total = 0
model.eval()
for img, label in tqdm(test_loader):

    # 입력 이미지 신경망 통과
    y=model(img.view(img.shape[0], -1))

    # 정확도 평가 기본 작업
    correct += (y.argmax(dim=1) == label).sum()
    total += label.numel()

# 학습 데이터셋에 대한 정확도 평가하기
acc = correct / total * 100
print(f"테스트 셋의 정확도는 {acc:.2f} 입니다.")