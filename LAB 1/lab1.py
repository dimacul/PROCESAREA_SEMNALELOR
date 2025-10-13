# Lab 1 -  Semnale

#200Hz = 200 ori/s

import matplotlib.pyplot as plt
import numpy as np
pi = np.pi

#-------------------------------------------------Exercitiul 1----------------------------------------------------

def x(t):
    return np.cos(520 * pi * t + pi / 3)
def y(t): 
    return np.cos(280 * pi * t - pi /3)
def z(t):
    return np.cos(120 * pi * t + pi /3)


#a) simulez axa reala de timp cu numere suficient de apropiate

axa_reala = np.arange(0, 0.03, 0.0005)

#b) 

fig, axs = plt.subplots(3)
fig.suptitle("Ploturi semnale")
axs[0].plot(axa_reala, x(axa_reala))
axs[1].plot(axa_reala, y(axa_reala))
axs[2].plot(axa_reala, z(axa_reala))

for ax in axs.flat:
    ax.set(xlabel='timp [s]', ylabel='amplitudine')
    
plt.savefig("Grafice/1b.pdf")
plt.show()

#c) esantionez cu o frecv de 200Hz ca sa obtin:
# - x[n]
# - y[n]
# - z[n]

f_esantionare = 200
#=>
d_esantioane_consec = 1/200

vector_t_esantionare = np.arange(0, 0.03, d_esantioane_consec)

fig, axs = plt.subplots(3)
fig.suptitle("Esantionari semnale")
axs[0].stem(vector_t_esantionare, x(vector_t_esantionare))
axs[1].stem(vector_t_esantionare, y(vector_t_esantionare))
axs[2].stem(vector_t_esantionare, z(vector_t_esantionare))

for ax in axs.flat:
    ax.set(xlabel='timp [s]', ylabel='amplitudine')
    
plt.savefig("Grafice/1c.pdf")
plt.show()


#-------------------------------------------------Exercitiul 2----------------------------------------------------

#a) f = 400Hz, 1600 esantioane

def a(t):
    return np.sin(2* pi * 400 * t)

f_esantionare_a = 1600 #=>
d_esantioane_consec_a = 1/1600

vector_t_esantionare_a = np.arange(0, 0.03, d_esantioane_consec_a)

plt.figure()
# plt.plot(axa_reala, a(axa_reala))
plt.stem(vector_t_esantionare_a, a(vector_t_esantionare_a))
plt.xlabel('timp [s]')
plt.ylabel('amplitudine')
plt.suptitle("Esantionare semnal a) f = 400Hz, f_esantionare=1600Hz")

plt.savefig("Grafice/2a.pdf")
plt.show()


#b) f = 800Hz, dureaza 3 secunde
def b(t):
    return np.sin(2* pi * 800 * t)

vector_t_esantionare_b = np.arange(0, 0.01, 0.00005)

plt.figure()
# plt.plot(axa_reala, a(axa_reala))
plt.plot(vector_t_esantionare_b, b(vector_t_esantionare_b))
plt.xlabel('timp [s]')
plt.ylabel('amplitudine')
plt.suptitle("Semnal b) f = 800Hz, dureaza 3 secunde")

plt.savefig("Grafice/2b.pdf")
plt.show()


#c) sawthooth, f = 240hz

def c(t):
    #return np.sin(2 * pi * 240 * t)
    return 2*pi*240*t - np.floor(2*pi*240 *t)

vector_t_esantionare_c = np.arange(0, 0.002, 0.0000005)

plt.figure()
# plt.plot(axa_reala, a(axa_reala))
plt.plot(vector_t_esantionare_c, c(vector_t_esantionare_c))
plt.xlabel('timp [s]')
plt.ylabel('amplitudine')
plt.suptitle("Semnal c) sawthooth, f = 240Hz")

plt.savefig("Grafice/2c.pdf")
plt.show()

#d) square, f = 300Hz

def d(t):
    return np.sign(np.sin(2*pi*300*t))

vector_t_esantionare_d = np.arange(0, 0.005, 0.0000005)

plt.figure()
# plt.plot(axa_reala, a(axa_reala))
plt.plot(vector_t_esantionare_d, d(vector_t_esantionare_d))
plt.xlabel('timp [s]')
plt.ylabel('amplitudine')
plt.suptitle("Semnal d) Square, f = 300Hz")

plt.savefig("Grafice/2d.pdf")
plt.show()


#e) semnal 2D aleator

array2Drandom = np.random.rand(128, 128)

print(array2Drandom)
plt.axis('off')

plt.savefig("Grafice/2e.pdf")
plt.imshow(array2Drandom)


#f) semnal 2D random, initializez cu o procedura creeata de mine
#0 = mov
array2Dregula = np.zeros((128,128))
for i in range(128):
    for j in range(128):
        if i+j == 128 or i == j:
            array2Dregula[i,j] = 1
        elif i<j:
            if i+j<128:
                array2Dregula[i,j] = 2
            else:
                array2Dregula[i,j] = 3
        elif i>j:
            if i+j<128:
                array2Dregula[i,j] = 4
            else:
                array2Dregula[i,j] = 5
        #     array2Dregula[i,j] = 3
plt.axis("Off")

plt.savefig("Grafice/2f.pdf")
plt.imshow(array2Dregula)

#-------------------------------------------------Exercitiul 3----------------------------------------------------

# Ex 3

# - semnal digitizat cu o frecventa de esant. 2000Hz

# a) interv de timp intre 2 esantioane?

# f_esantionare = 2000Hz = 2000 esantionari pe secunda

# => dist_2_esant_consec = 1/f_esantionare = 1/2000 = 0.0005 s


# b) un esantion e mem. pe 4b

# 1 ora achizitie = 3600 s

# nr_esantioane = 3600 s * 2000Hz = 7200000

# total_mem_ocupata = 7200000 * 4b = 28800000 b

# 1byte = 8b=>
# in bytes: total_mem_ocupata = 28800000 / 8 = 3600000 = 36*10^5 B
