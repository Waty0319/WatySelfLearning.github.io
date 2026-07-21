import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

# generating data

N = 10
#number of samples for model training
N_test = 100
#number of samples for calculationg test error

rng1 = np.random.default_rng(0)
rng2 = np.random.default_rng(1)
#random number generator reset

x = np.linspace(0,1,N)
#training data -> 10 uniformly spaced points on [0,1]
x_test = np.linspace(0,1,N_test)
#test data -> 100 uniformly spaced points on [0,1] 

noise = rng1.normal(loc=0, scale=0.25, size=N)
#10 noise samples for training data following N(0,0.25^2)
noise_test = rng2.normal(loc=0,scale=0.25,size=N_test)
#100 noise samples for test data following N(0,0.25^2)

t = np.sin(2*np.pi*(x)) + noise
#generating 10 training target 
t_test = np.sin(2*(np.pi)*(x_test)) + noise_test
#generating 100 test target

# 2. basic funtions

def fit_polynomial(x,t,M) :
#returns the coefficient vector W of the best-fit M-th order polynomial

    A = np.zeros((M+1,M+1))
    T = np.zeros(M+1)
    #initialize matrix A and vector T with zeros

    for i in range(M+1):
        T[i] = np.sum((x**i)*t)
        for j in range(M+1):
            A[i,j] = np.sum(x**(i+j))
    #populate the entries of A and T

    W = np.linalg.solve(A,T)
    #solving Aw = T

    return W

def fit_polynomial_regulated(x,t,M,lam) :
#returns the coefficient vector W of the best-fit M-th order polynomial under regularization

    A = np.zeros((M+1,M+1))
    T = np.zeros(M+1)

    for i in range(M+1):
        T[i] = np.sum((x**i)*t)
        for j in range(M+1):
            A[i,j] = np.sum(x**(i+j))

    W = np.linalg.solve(A+lam*np.eye(M+1),T)

    return W

def sum_of_square_error(x,t,N,M,W) :
#returns value of RMS error

    temp = 0
    #intermediate value
    error_sum = 0
    #initialize sum of square error
        
    for j in range(N):
        for i in range(M+1):
            temp += (W[i]*(x[j]**i))
        error_sum += (temp - t[j])**2
        temp = 0
                
    error = np.sqrt(error_sum/N)

    return error

#3.illustrates plot

M_list = [0, 1, 3, 9]
x_curve = np.linspace(0,1,1000)
y_true = np.sin(2*np.pi*(x_curve))

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 8))

axes_flat = axes.flat
#simplify iteraion by flating axes

for M, ax in zip(M_list, axes_flat):
    
    W = fit_polynomial(x, t, M)
    e_test = sum_of_square_error(x_test,t_test,N_test,M,W)
    e_train = sum_of_square_error(x,t,N,M,W)

    y_curve = np.zeros_like(x_curve)
    for i in range(M + 1):
        y_curve += W[i] * (x_curve ** i)
    #generating fit curve
        
    ax.scatter(x, t, facecolor='none', edgecolor='blue', s=50, label='Data')
    #points data(feature & target)
    
    ax.plot(x_curve, y_true, color='green', label='True')
    #true curve

    ax.plot(x_curve, y_curve, color='red', label=f'Fit')
    #fit curve

    ax.set_title(f"M = {M} \n, E_train = {round(e_train,3)}, E_test = {round(e_test,3)}")
    #title

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-1.5, 1.5)
    #area of drawing

    ax.legend()
    ax.grid(True)

#drawing fit polynomial
W = fit_polynomial_regulated(x, t, 9 , 0.0005)
e_test = sum_of_square_error(x_test,t_test,N_test,9,W)
e_train = sum_of_square_error(x,t,N,9,W)

y_curve = np.zeros_like(x_curve)
for i in range(10):
    y_curve += W[i] * (x_curve ** i)

axes_flat[4].scatter(x, t, facecolor='none', edgecolor='blue', s=50, label='Data')
axes_flat[4].plot(x_curve, y_true, color='green', label='True')
axes_flat[4].plot(x_curve, y_curve, color='red', label=f'Fit')
    
axes_flat[4].set_title(f"lambda = 0.0005, M = 9 \n E_train = {round(e_train,3)} E_test = {round(e_test,3)}")
axes_flat[4].set_xlim(-0.1, 1.1)
axes_flat[4].set_ylim(-1.5, 1.5)
axes_flat[4].legend()
axes_flat[4].grid(True)

#
M_all = np.arange(10)
#order of polynomial
lam_all = [0.5 , 0.05 , 0.005 , 0.0005, 0.00005, 0.000005 , 0.0000005, 0.00000005]
#regularizaion strength
e1 = np.zeros(10)
#test error
e2 = np.zeros(10)
#training error
e_regulated1 = np.zeros(8)
#test error under regularization
e_regulated2 = np.zeros(8)
#training error under regularization

for element in M_all :
    W = fit_polynomial(x, t, element)
    e1[element] = sum_of_square_error(x_test,t_test,N_test,element,W)
    e2[element] = sum_of_square_error(x,t,N,element,W)
#error versus order M

for i in range(8) :
    W = fit_polynomial_regulated(x,t,9,lam_all[i])
    e_regulated1[i] = sum_of_square_error(x_test,t_test,N_test,9,W)
    e_regulated2[i] = sum_of_square_error(x,t,N,9,W)
#error vesus regulariztion strength 

axes_flat[5].plot(np.arange(len(e1)), e1, color='red', label='test error M=0~9')
axes_flat[5].plot(np.arange(len(e2)), e2, color='blue', label='train error M=0~9')
axes_flat[6].plot(np.arange(len(e_regulated1)), e_regulated1, color='green', label='regulated M=9 test error \n lambda = 5e-(1~8)')
axes_flat[6].plot(np.arange(len(e_regulated2)), e_regulated2, color='black', label='regulated M=9 training error \n lambda = 5e-(1~8)')

axes_flat[5].set_xlim(-0.1, 10.1)
axes_flat[5].set_ylim(-1, 1)
axes_flat[6].set_xlim(-0.1, 8.1)
axes_flat[6].set_ylim(-0.5, 0.5)

axes_flat[5].legend()
axes_flat[6].legend()
axes_flat[5].grid(True)
axes_flat[6].grid(True)

axes_flat[7].axis('off')
#hide unused flat

plt.tight_layout()
plt.show()

#책의 예시를 그대로 적용하면 소수점이 너무 작아서 인식하지 못하는 문제가 생김 => 오버피팅이 재현되지 않는다.
#내 아이디어 => 구간 크기를 몇배로 늘리면 소수점이 인식될 것이다.
#여전히 문제가 생김 -> 오직 데이터의 분산을 조정할 때 오버피팅이 관찰됨 -> 분산을 조절하자.
#분산과 구간 스케일을 함께 조절하면 그대로 -> 데이터 사이즈는 조절해보지 않았다.
#M을 41로 해보고 나서야 오버피팅이 재현되었다. -> M=9는 오버피팅을 재현하기에 너무 낮은 수치였다.
#의심 : 오버핏을 직관적으로 판단하고 있는 것 아닐까? 오버핏해보이지 않지만 실제로 오버핏할수도? -> 예측오차 도입하자.
#예측오차가 M=1 < M=3 로 나타난다. 왜? -> W_0의 값이 계속 더해진다. -> 1~M으로 더하기 -> 어짜피 W_0 아주 작아서 영향 미미... 원인 아님
#진짜 원인 : rms의 구조적 결함, t_test과의 차이가 다항함수를 출력하는 반복문안에 들어가서 계속 마이너스되는 구조 -> 해결
#M=9일 때 rms가 더 높게 나타남. 극적인 차이는 아니지만 더 크다는게 핵심 -> 오버피팅의 증거
#정규화항 도입 -> 기존 식을 homogeneous eq로 보고, 정규화항을 포함하여 편미분한 식은 nonhomogeneous eq로 보고 정규화항 없는 식 재활용하여 유도
# -> A+lamI=T를 푼다.
#규제항 도입 -> 오버피팅 개선 + 기존 최적이라 알려진 M=3보다 성능 좋음 -> 자유도 높이고 규제항 도입하는게 규제항 없이 자유도만 줄이는 것보다 나을 수 있다.
#영어 공부 겸 주석 영어로 직접 추가
