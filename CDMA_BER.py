#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
c1=[1,1,1,1];
c2=[1,-1,1,-1];
c3=[1,1,-1,-1];
c4=[1,-1,-1,1];
rc=[];
print("Enter the data bits :") 
d1=int(input("Enter D1 :")) 
d2=int(input("Enter D2 :")) 
d3=int(input("Enter D3 :")) 
d4=int(input("Enter D4 :")) 
r1=np.multiply(c1,d1) 
r2=np.multiply(c2,d2) 
r3=np.multiply(c3,d3) 
r4=np.multiply(c4,d4) 
resultant_channel=r1+r2+r3+r4; 
print("Resultant Channel",resultant_channel) 
Channel=int(input("Enter the station to listen for C1=1 ,C2=2, C3=3 C4=4 : ")) 
if Channel==1: 
    rc=c1 
elif Channel==2: 
    rc=c2 
elif Channel==3: 
    rc=c3 
elif Channel==4: 
    rc=c4 
inner_product=np.multiply(resultant_channel,rc) 
print("Inner Product",inner_product) 
res1=sum(inner_product) 
data=res1/len(inner_product) 
print("Data bit that was sent",data) 


# In[2]:


# There are several types of wireless communication channels, and two common examples a
# (AWGN) channel and the Rayleigh fading channel.
# AWGN channel: In this type of channel, the received signal is corrupted by additive w
 # which is a type of random noise that is characterized by its mean and variance.
 # The AWGN channel is often used to model wireless communication channels
 # where the signal experiences random noise due to factors such as atmospheric inte
 # electronic noise, and thermal noise.
# Rayleigh fading channel: In this type of channel, the signal experiences random varia
# due to multiple signal paths between the transmitter and receiver.
# This effect is known as multipath fading, and it can result in signal distortion and
# The Rayleigh fading channel is commonly used to model wireless communication channels
# in urban or indoor environments where there are many obstructions that cause signal r
# Other types of wireless communication channels include:
# Rician fading channel: This is a variation of the Rayleigh fading channel that includ
# a dominant line-of-sight path in addition to the scattered paths.
# Nakagami-m fading channel: This is a generalization of the Rayleigh fading channel th
# a parameter m that determines the severity of the fading.
# Flat fading channel: In this type of channel, the signal experiences a constant atten
# over the entire bandwidth of the signal.
# Frequency selective fading channel: In this type of channel,
# different frequency components of the signal experience different levels of attenuati
# resulting in distortion of the signal waveform.
# The Bit Error Rate (BER) performance over a Rayleigh fading channel
# with Binary Phase Shift Keying (BPSK) transmission can be analyzed using the followin
# BER = 0.5 * erfc(np.sqrt(10 ** (Eb_N0_dB / 10)))
# where erfc is the complementary error function and SNR is the Signal-to-Noise Ratio.
# To simulate the BER performance over a range of SNR values from 0 to 60 dB, we can us
import numpy as np
from scipy.special import erfc
import matplotlib.pyplot as plt
# Parameters
N = int(1e6) #number of bits or symbols
Eb_N0_dB = np.arange(-3, 60) #multiple Eb/N0 (SNR) values
# Transmitter
ip = np.random.rand(N) > 0.5 #generating 0,1 with equal probability
s = 2*ip - 1 #BPSK modulation 0 -> -1; 1 -> 1
# Simulation
nErr = np.zeros(len(Eb_N0_dB))
for i, Eb_N0 in enumerate(Eb_N0_dB):
 n = np.sqrt(0.5) * (np.random.randn(N) + 1j*np.random.randn(N)) #white gaussian noise
 h = np.sqrt(0.5) * (np.random.randn(N) + 1j*np.random.randn(N)) # Rayleigh fading channel
 y = h*s + np.sqrt(10**(-Eb_N0/10))*n #received Signal
 ipHat = (np.real(y/h) > 0).astype(int) #receiver - hard decision decoding
 nErr[i] = np.sum(ip != ipHat)
# BER calculation
simBer = nErr / N
theoryBerAWGN = 0.5*erfc(np.sqrt(10**(Eb_N0_dB/10)))
theoryBer = 0.5*(1 - np.sqrt(10**(Eb_N0_dB/10) / (1 + 10**(Eb_N0_dB/10))))
# Plot
plt.semilogy(Eb_N0_dB, theoryBerAWGN, 'cd-', linewidth=2)
plt.semilogy(Eb_N0_dB, theoryBer, 'bp-', linewidth=2)
plt.semilogy(Eb_N0_dB, simBer, 'mx-', linewidth=2)
plt.axis([-3, 35, 1e-5, 0.5])
plt.grid(True, which="both")
plt.legend(['AWGN-Theory', 'Rayleigh-Theory', 'Rayleigh-Simulation'])
plt.xlabel('Eb/No, dB')
plt.ylabel('Bit Error Rate')
plt.title('BER for BPSK modulation in Rayleigh channel')
plt.show()


# In[4]:


print(Eb_N0_dB)


# In[13]:


import numpy as np
import matplotlib.pyplot as plt
# Set the parameters
n_bits = 1000000 # Number of bits to be transmitted
SNRdBs = np.arange(-10, 11, 1) # SNR range in dB
SNRs = 10**(SNRdBs/10) # SNR range in linear scale
# Generate random bits
bits = np.random.randint(0, 2, n_bits)
# Loop over SNR values
BERs = []
for SNR in SNRs:
 # BPSK modulation
 symbols = 2*bits - 1
 # Add noise
 noise_power = 1/SNR
 noise = np.sqrt(noise_power)*np.random.randn(n_bits)
 received = symbols + noise
 # BPSK demodulation
 decoded_bits = (received >= 0).astype(int)
 # Calculate the bit error rate
 BER = np.sum(bits != decoded_bits) / n_bits
 BERs.append(BER)
# Plot the results
plt.semilogy(SNRdBs, BERs)
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate')
plt.title('Bit Error Rate vs. SNR for BPSK modulation with AWGN')
plt.grid(True)
plt.show()


# In[ ]:




