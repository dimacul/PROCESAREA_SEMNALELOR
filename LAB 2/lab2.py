import scipy.io.wavfile
import scipy.signal 
import sounddevice
import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------- Ex. 1 -------------------------------

#1. cos care arata ca sin

# sin(t) = cos(t - pi/2)
# cos(t) = sin(t + pi/2)

A = 1
f = 5
faza = np.pi/6 #cu cat dau valori mai mici pt faza, cu atat semnalul sin se deplaseaza spre dreapta mai mult
axa_reala = np.arange(0, 1, 0.0005)


def semnal_sinusoidal_sinus(A, f, faza, t):
    return A * np.sin(2 * np.pi * f * t + faza)

def cos_care_arata_ca_sin(A, f, faza, t):
    return A * np.cos(2 * np.pi * f * t + faza - np.pi/2)

fig, axs = plt.subplots(2, 1, figsize=(10,10))
fig.suptitle("Semnal sin si semnal cos care arata identic cu sin \n sin(t) = cos(t - pi/2)")
axs[0].plot(axa_reala, semnal_sinusoidal_sinus(A, f, faza, axa_reala))
axs[0].set_title('Semnal sinusoidal de tip sinus | sin(t) = sin(2π*5*t + pi/6)')

axs[1].plot(axa_reala, cos_care_arata_ca_sin(A, f, faza, axa_reala))
axs[1].set_xlabel("timp [s]")
axs[1].set_title('Semnal sinusoidal de tip cosinus care arata ca sin | cos(t) = cos(2π*5*t + pi/6 - pi/2)')

for ax in axs.flat:
    ax.set( ylabel='amplitudine')

plt.savefig("1_Semnale_sinusoidale_sin_si_cos_care_arata_ca_sin.pdf")
plt.show()


#-------------------------------------- Ex. 2 -------------------------------

#2. semnal sinusoidal de tip sinus, amplitudine = 1, frecv = ce vreau eu, incerc 4 valori dif pt faza, toate semnalele in acelasi plot

def semnal_sinusoidal_sinus(A, f, faza, t):
    return A * np.sin(2 * np.pi * f * t + faza)

A = 1
f = 5
faze = [0, np.pi/6, np.pi/4, np.pi/3]

axa_reala = np.arange(0, 1, 0.0005)


fig, axs = plt.subplots(4, figsize = (8, 12)) #sharex=True
fig.suptitle("Semnale sin de A = 1 si faze diferite (0, pi/6, pi/4, pi/3)\nCu cat faza e mai mare, cu atat semnalul vine mai spre stanga")
axs[0].plot(axa_reala, semnal_sinusoidal_sinus(A, f, faze[0], axa_reala))
axs[0].set_title('sin(t) = sin(2pi*5*t + 0)')

axs[1].plot(axa_reala, semnal_sinusoidal_sinus(A, f, faze[1], axa_reala))
axs[1].set_title('sin(t) = sin(2pi*5*t + pi/6)')

axs[2].plot(axa_reala, semnal_sinusoidal_sinus(A, f, faze[2], axa_reala))
axs[2].set_title('sin(t) = sin(2pi*5*t + pi/4)')

axs[3].plot(axa_reala, semnal_sinusoidal_sinus(A, f, faze[3], axa_reala))
axs[3].set_title('sin(t) = sin(2pi*5*t + pi/3)')
axs[3].set_xlabel("timp [s]")

for ax in axs.flat:
    ax.set( ylabel='amplitudine')
    
plt.tight_layout(rect=[0, 0, 1, 1]) #colt stg jos -> dr sus available pt plotare

plt.savefig("2_Semnale sin - 4 valori diferite pentru faza.pdf")
plt.show()

# Pt unul din semnalele anterioare, adaug zgomot aleator sinusoidei esantionate

# Noul semnal:

# x[n]+gamma*z[n]

# SNR =         ||x||^2

#                -------

#             gamma^2 * ||z||^2

# SNR sa fie 0.1, 1, 10, 100

# z - il obtin esantionand distrib gaussiana std

# gamma = ||x|| / (  ||z|| * sqrt(SNR)  )



#esantionez o sinusoida de la ex anterior:

A = 1
f = 5
faza = faze[0] #0 grade

x = semnal_sinusoidal_sinus(A, f, faza, axa_reala)
# print(x)
# print(x.shape)

z = np.random.normal(loc = 0.0, scale = 1, size = x.shape)
# print(z)
# print(z.shape)

SNR = [0.1, 1, 10, 100] 

gammas = np.zeros(len(SNR))
for i in range(len(SNR)):
    gammas[i] =  np.linalg.norm(x) / (np.sqrt(SNR[i]) * np.linalg.norm(z))
    
# print("Gammas: ", gammas)
# print("x: ", x)
# print("z: ", z)

fig, axs = plt.subplots(4, figsize = (10, 12)) #sharex=True
fig.suptitle("Semnale esantionate care contin zgomot gaussian x[n] + gamma*z[n], cu diferite valori pentru SNR")
axs[0].plot(axa_reala, x + gammas[0]*z )
axs[1].plot(axa_reala, x + gammas[1]*z )
axs[2].plot(axa_reala, x + gammas[2]*z )
axs[3].plot(axa_reala, semnal_sinusoidal_sinus(A, f, faza, axa_reala) + gammas[3]*z )
# axs[3].plot(vector_t_esantionare, semnal_sinusoidal_sinus(A, f, faza, axa_reala))

for i, ax in enumerate(axs.flat):
    ax.set( ylabel='amplitudine', title='SNR = {}'.format(SNR[axs.tolist().index(ax)]))
    
plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig("2_Semnale sin - 4 valori diferite SNR.pdf")
plt.show()


#-------------------------------------- Ex. 3 -------------------------------

# Ascult semnalele de la lab precedent pt 2a -> f cu sounddevice.

# Salvez unul ca .wav + verific ca il pot incarca de pe disc cu scipy.io.wavfile.read().

#Din labul trecut:
#2a)

def a(t):
    return np.sin(2* np.pi * 400 * t)

signal_a = np.array(a(np.arange(0, 5, 1/44100)))

#salvare semnal in format audio .wav
rate = int(10e5)
scipy.io.wavfile.write("3_signal2a", rate, signal_a)

#redare
fs = 44100 #frecv de esantionare
sounddevice.play(signal_a, fs)


#2b)

def b(t):
    return np.sin(2* np.pi * 800 * t)
signal_b = np.array(b(np.arange(0, 5, 1/fs)))

sounddevice.play(signal_b, fs)

# frecvență mică → sunet gros / grav
# frecvență mare → sunet subțire / ascuțit


#2c)

def c(t): #sawthooth
    return 2*np.pi*1*t - np.floor(2*np.pi*1 *t)

signal_c = np.array(c(np.arange(0, 5, 1/fs)))
sounddevice.play(signal_c, fs)


#2d)

def d(t): #square
    return np.sign(np.sin(2*np.pi*1*t))

signal_d = np.array(d(np.arange(0, 5, 1/fs)))
sounddevice.play(signal_d, fs)


#-------------------------------------- Ex. 4 -------------------------------

# 2 semnale cu forme de unda diferite (ex: sawthooth, sinusoidal)

# le adun esantioanele

# Afisez grafic:
# - semnal 1
# - semnal 2
# - suma lor

# //fiecare in cate un subplot

def sawtooth(t): #sawthooth
    return 2*np.pi*1*t - np.floor(2*np.pi*1*t)

def sinusoidal(t):
    return np.sin(2*np.pi*1*t)

axa_reala = np.arange(0, 1, 0.0005)

sawtooth = sawtooth(axa_reala)
sinusoidal = sinusoidal(axa_reala)

fig, axs = plt.subplots(3, figsize = (10, 12)) #sharex=True
fig.suptitle("Doua semnale cu forma de unda diferite si suma lor")
axs[0].plot(axa_reala, sawtooth )
axs[0].set_title('Semnal cu forma de unda sawtooth')

axs[1].plot(axa_reala, sinusoidal )
axs[1].set_title('Semnal cu forma de unda sinusoidala')

axs[2].plot(axa_reala, sawtooth + sinusoidal )
axs[2].set_title('Suma celor doua semnale')
axs[2].set_xlabel("timp [s]")


# for i, ax in enumerate(axs.flat):
#     ax.set( ylabel='amplitudine', title='SNR = {}'.format(SNR[axs.tolist().index(ax)]))
    
plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig("4_sin + sawtooth + suma_lor.pdf")
plt.show()


#-------------------------------------- Ex. 5 -------------------------------

5.

# Doua semnale cu aceeasi forma de unda, DAR frecv diferite.

# Unul dupa celalalt intr-un vector.

# Redau audio + obs.

def sin1(t):
    return np.sin(2 * np.pi * 2 * t)

def sin2(t):
    return np.sin(2 * np.pi * 4 * t)

sin1 = sin1(axa_reala)
sin2 = sin2(axa_reala)

concatenate = np.concatenate((sin1, sin2))
print(sin1.shape, sin2.shape, concatenate.shape)

axa_reala_concat = np.arange(0, 2, 0.0005)


plt.title("Doua semnale cu aceeasi forma de unda, dar frecv. diferite (1Hz / 2Hz), concatenate")
plt.plot(axa_reala_concat, concatenate)
plt.xlabel("timp [s]")
plt.ylabel("amplitudine")

plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig("5_doua_sinusoide_de_frecv_dif_concatenate.pdf")
plt.show()

#daca le ascult pe astea nu se aude aproape nimic, frecvente prea joase



fs = 44100 #frecv de esantionare

def sin1(t):
    return np.sin(2 * np.pi * 400 * t)

def sin2(t):
    return np.sin(2 * np.pi * 800 * t)

sinus1 = sin1(np.arange(0, 3, 1/fs))
sinus2 = sin2(np.arange(0, 3, 1/fs))

concatenate = np.concatenate((sinus1, sinus2))



sounddevice.play(concatenate, fs)

# frecventa mica => sunet jos
# frecventa mare => sunet inalt, asccutit

#salvare semnal in format audio .wav
rate = int(10e5)
scipy.io.wavfile.write("5_2semnale_frecv_dif_concatenate", rate, concatenate)


#-------------------------------------- Ex. 6 -------------------------------

# 3 semnale sin, A = 1, faza = 0,

# - f = fs / 2
# - f = fs / 4
# - f = 0Hz

fs = 44100 #frecv de esantionare

def sin_a(t):
    return np.sin(2 * np.pi * (fs/2) * t)

def sin_b(t):
    return np.sin(2 * np.pi * (fs/4) * t)

def sin_c(t):
    return np.sin(2 * np.pi * 0 * t)

sinus_a = sin_a(np.arange(0, 3, 1/fs))
sinus_b = sin_b(np.arange(0, 3, 1/fs))
sinus_c = sin_c(np.arange(0, 3, 1/fs))

#sounddevice.play(sinus_a, fs) #nu aud, frecventa prea inalta
# sounddevice.play(sinus_b, fs) #doar pe asta o aud
sounddevice.play(sinus_c, fs) #asta oricum e mereu 0

# concatenate_abc = np.concatenate((sinus_a, sinus_b, sinus_c))
# sounddevice.play(concatenate_abc, fs)


#--------------------------------

fs = 44100
def sin_a(t):
    return np.sin(2 * np.pi * (fs/2) * t)

fig, axs = plt.subplots(2, figsize = (10, 12)) #sharex=True
fig.suptitle("sin_a(t) = sin(2π*(fs/2)*t) si esantionarea sa cu frecventa fs")
axs[0].plot(np.arange(0, 0.001, 0.000005), sin_a(np.arange(0, 0.001, 0.000005)) )
axs[0].set_title('sin(t) = sin(2π*(fs/2)*t)')

axs[1].stem(np.arange(0, 0.001, 1/fs), sin_b(np.arange(0, 0.001, 1/fs)) )
axs[1].set_title('Esantionare sin(t) = sin(2π*(fs/2)*t)')

axs[1].set_xlabel("timp [s]")

plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig("6_a.pdf")
plt.show()

#--------------------------------

fs = 44100
def sin_b(t):
    return np.sin(2 * np.pi * (fs/4) * t)

fig, axs = plt.subplots(2, figsize = (10, 12)) #sharex=True
fig.suptitle("sin_b(t) = sin(2π*(fs/2)*t) si esantionarea sa cu frecventa fs")
axs[0].plot(np.arange(0, 0.001, 0.000005), sin_b(np.arange(0, 0.001, 0.000005)) )
axs[0].set_title('sin(t) = sin(2π*(fs/4)*t)')

axs[1].stem(np.arange(0, 0.001, 1/fs), sin_b(np.arange(0, 0.001, 1/fs)) )
axs[1].set_title('Esantionare sin(t) = sin(2π*(fs/4)*t)')

axs[1].set_xlabel("timp [s]")

plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig("6_b.pdf")
plt.show()

#--------------------------------

fs = 44100
def sin_c(t):
    return np.sin(2 * np.pi * 0 * t)

fig, axs = plt.subplots(2, figsize = (10, 12)) #sharex=True
fig.suptitle("sin_c(t) = sin(2π*0*t) si esantionarea sa cu frecventa fs")
axs[0].plot(np.arange(0, 0.001, 0.000005), sin_c(np.arange(0, 0.001, 0.000005)) )
axs[0].set_title('sin(t) = sin(2π*0*t)')

axs[1].stem(np.arange(0, 0.001, 1/fs), sin_c(np.arange(0, 0.001, 1/fs)) )
axs[1].set_title('Esantionare sin(t) = sin(2π*0*t)')

axs[1].set_xlabel("timp [s]")

plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig("6_c.pdf")
plt.show()

# Observatii:

# Pt sinusoidele f = fs / 2 si f = fs / 4: Cand le esantionez, par aceeasi sinusoida.

# Exista o inf. de sinusoide care ar putea da o anumita esantionare.

# Doar sinusoida f = fs/4 se aude cu sounddevice. Prima are frecv prea mare, iar ultima e 0.

#-------------------------------------- Ex. 7 -------------------------------

# semnal sinusoidal fs = 1000 Hz
# - decimez la 1/4 din frecv initiala (pastrez doar al patrulea fiecare elem din vector)

# a) afisez grafic cele 2 semnale + comentez diferentele
# b) repet decimarea tot la 1/4 din frecv initiala, dar pornesc de la al doilea element din vector

fs = 1000  # frecventa de esantionare

def sin_ex_7(t):
    return np.sin(2 * np.pi * 150 * t)

vector_esantioane = np.arange(0, 0.07, 1/fs)
vector_esantioane_decimat = vector_esantioane[::4] # sau dist dintre doua esantioane e 4 ori mai mare =   1   /   fs/4

fig, axs = plt.subplots(2, figsize = (10, 12)) #sharex=True
fig.suptitle("sin_ex_7(t) = sin(2π*150*t) - esantionat, apoi esantionat de 4 ori mai rar")
axs[0].stem(vector_esantioane, sin_ex_7(vector_esantioane), markerfmt='black' )
axs[0].set_title('sin(t) = sin(2π*150*t) - esantionat la fs = 1000Hz')
axs[0].set_ylabel("amplitudine")
axs[1].stem(vector_esantioane_decimat, sin_ex_7(vector_esantioane_decimat) , markerfmt='black')
axs[1].set_title('sin(t) = sin(2π*150*t) - esantionat la fs = 250Hz (decimat de 4 ori)')

axs[1].set_xlabel("timp [s]")
axs[1].set_ylabel("amplitudine")

plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig("7_a.pdf")
plt.show()

#------------------------------

fs = 1000  # frecventa de esantionare

def sin_ex_7(t):
    return np.sin(2 * np.pi * 150 * t)

vector_esantioane = np.arange(0, 0.07, 1/fs)
vector_esantioane_decimat = vector_esantioane[1::4] # sau dist dintre doua esantioane e 4 ori mai mare =   1   /   fs/4

fig, axs = plt.subplots(2, figsize = (10, 12)) #sharex=True
fig.suptitle("sin_ex_7(t) = sin(2π*150*t) - esantionat, apoi esantionat de 4 ori mai rar")
axs[0].stem(vector_esantioane, sin_ex_7(vector_esantioane), markerfmt='black' )
axs[0].set_title('sin(t) = sin(2π*150*t) - esantionat la fs = 1000Hz')
axs[0].set_ylabel("amplitudine")
axs[1].stem(vector_esantioane_decimat, sin_ex_7(vector_esantioane_decimat) , markerfmt='black')
axs[1].set_title('sin(t) = sin(2π*150*t) - esantionat la fs = 250Hz, dar pornind de la al doilea elem. din vector')

axs[1].set_xlabel("timp [s]")
axs[1].set_ylabel("amplitudine")

plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig("7_b.pdf")
plt.show()

# Obs: 
# Pentru cele 2 decimari, esantioanele arata diferit.
# Pentru a doua esantionare, semnalul e si mai greu de prezis decta la prima.


#-------------------------------------- Ex. 8 -------------------------------

# Pt valori mici alpha:

# sin(alpha) = alpha

# - E aproximarea corecta? Reprez. grafic cele 2 curbe pentru valori ale lui alpha in [-pi/2, pi/2]
# - Grafic cu eroarea dintre cele 2 functii.


#------------------------------------TAYLOR
def sin(alpha):
    return np.sin(alpha)

def alpha_function(alpha):
    return alpha

axa_reala = np.arange(-np.pi/2, np.pi/2, 0.005)

fig, axs = plt.subplots(3, figsize = (10, 18)) #sharex=True
# fig.suptitle("sin(alpha) vs alpha")
axs[0].plot(axa_reala, sin(axa_reala), color = "blue", label = "sin(alpha)")
axs[0].plot(axa_reala, alpha_function(axa_reala), color = "red", label = "alpha")
axs[0].set_title('sin(alpha) & alpha')
axs[0].set_xlabel("alpha")
axs[0].set_ylabel("Valoarea functiei")
axs[0].legend()

def eroare(alpha):
    return abs(sin(alpha) - alpha)

axs[1].plot(axa_reala, eroare(axa_reala))
axs[1].set_title('Eroarea dintre cele 2 functii: abs(sin(alpha) - alpha)')

axs[1].set_xlabel("alpha")
axs[1].set_ylabel("Valoarea erorii")

axs[2].plot(axa_reala, eroare(axa_reala))
axs[2].set_title('Eroarea - Oy logaritmic')

axs[2].set_xlabel("alpha")
axs[2].set_ylabel("Valoarea erorii (logaritmic)")
axs[2].set_yscale("log")


plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig("8_Taylor.pdf")
plt.show()


#------------------------------------PADE

# Aproximarea Pade:
# sin(α)≈ α − 7α^3/60 / (1+α^2/20)

def aprox_pade(alpha):
    return (alpha - 7 * ((alpha**3)/60)) / (1 + (alpha**2/20))

axa_reala = np.arange(-np.pi/2, np.pi/2, 0.005)

fig, axs = plt.subplots(3, figsize = (10, 16)) #sharex=True
# fig.suptitle("sin(alpha) vs alpha")
axs[0].plot(axa_reala, sin(axa_reala), color = "blue", label = "sin(alpha)")
axs[0].plot(axa_reala, aprox_pade(axa_reala), linestyle='dotted', color = "red", label = "aprox_pade(alpha)")
axs[0].set_title('sin(alpha) & aproximarea Pade pentru sin(alpha)')
axs[0].set_xlabel("alpha")
axs[0].set_ylabel("Valoarea functiei")
axs[0].legend()

def eroare_pade(alpha):
    return abs(sin(alpha) - aprox_pade(alpha))

axs[1].plot(axa_reala, eroare_pade(axa_reala))
axs[1].set_title('Eroarea dintre cele 2 functii: abs(sin(alpha) - aprox_pade(alpha)')

axs[1].set_xlabel("alpha")
axs[1].set_ylabel("Valoarea erorii")

#eroarea pe scara logaritmica
axs[2].plot(axa_reala, eroare_pade(axa_reala))
axs[2].set_title('Eroarea - Oy logaritmic')

axs[2].set_xlabel("alpha")
axs[2].set_ylabel("Valoarea erorii (logaritmic)")
axs[2].set_yscale("log")


plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig("8_Pade.pdf")
plt.show()