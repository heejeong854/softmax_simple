import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import matplotlib.font_manager as fm
import platform

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')  # 맑은 고딕
elif platform.system() == 'Darwin':  # macOS
    plt.rc('font', family='AppleGothic')
else:  # Linux (예: Streamlit Cloud)
    plt.rc('font', family='NanumGothic')

# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# 모델 정의
class SimpleNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28*28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)

# Softmax 함수
def softmax(logits):
    exps = torch.exp(logits - torch.max(logits))
    return exps / torch.sum(exps)

# 0~9 숫자가 하나씩 포함된 10개 샘플만 선택
def get_balanced_subset(dataset):
    selected = []
    found_labels = set()
    for img, lbl in dataset:
        if lbl not in found_labels:
            selected.append((img, lbl))
            found_labels.add(lbl)
        if len(found_labels) == 10:
            break
    return selected

# 제목 및 설명
st.title("🧠 Softmax 숫자 인식 시뮬레이터")
st.markdown("MNIST 이미지 중 0~9 숫자를 하나씩 뽑아 모델 예측과 Softmax 분포를 확인해보세요.")
st.header("Softmax는 logit → 확률 분포로 바꿔주는 함수")
st.write("logit을 지수함수로 정규화하여 확률 분포로 만듭니다. 확인해봅시다!")

# 이미지 transform 정의
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# MNIST 테스트셋 로드
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 10개 숫자 추출
balanced_samples = get_balanced_subset(mnist_test)

# 슬라이더로 이미지 선택
index = st.slider("이미지 번호 (0~9)", 0, 9, 0)
image, label = balanced_samples[index]

# 정규화 해제해서 시각화용 이미지 생성
image_for_display = image * 0.3081 + 0.1307
st.image(image_for_display.squeeze().numpy(), caption=f"Ground Truth: {label}", width=150)

# 모델 정의 및 추론
model = SimpleNN()
# 학습된 모델 불러오기 예시
model.load_state_dict(torch.load("trained_mnist_model.pth"))

model.eval()

with torch.no_grad():
    logits = model(image.unsqueeze(0)).squeeze()
    probs = softmax(logits).numpy()
    logits_np = logits.numpy()

predicted = int(np.argmax(probs))
st.write(f"### ✅ 예측 결과: {predicted}")

# 로짓 그래프
st.subheader("📊 Logits (모델 출력값)")
fig1, ax1 = plt.subplots()
ax1.bar(range(10), logits_np, color='gray')
ax1.set_xticks(range(10))
ax1.set_xlabel("클래스 (0~9)")
ax1.set_ylabel("로짓 값")
st.pyplot(fig1)

# Softmax 확률 분포 그래프
st.subheader("📈 Softmax 확률 분포")
fig2, ax2 = plt.subplots()
bars = ax2.bar(range(10), probs, color='skyblue')
bars[predicted].set_color('orange')
ax2.set_xticks(range(10))
ax2.set_xlabel("클래스 (0~9)")
ax2.set_ylabel("확률")
ax2.set_ylim(0, 1)
st.pyplot(fig2)
