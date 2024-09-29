import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt 

"""
-----------------------------------------------------------------------------------------------------------------------------------
#For step 1, we have to pull out all the columns in the cepheid data sheet.
We will obtain the absolute mag M and propagate the errors on it.
We will also discorrelate the intercept and gradient.
-----------------------------------------------------------------------------------------------------------------------------------
"""
par, par_error, period, vis_mag, A_extinction, A_error = np.loadtxt('MW_Cepheids.dat', dtype='float', usecols=(1,2,3,4,5,6), unpack=True)

def step1(parallax, parallax_error, periods, m_values, A, A_err):
    relative_distance = 1000/parallax #this gives the relative distance
    log_d = np.log10(relative_distance)
    M = m_values - 5*log_d + 5 - A #these are the m values
    log_period = np.log10(periods)
    shifted_logP = log_period - np.mean(log_period) #shifted logP values to discorrelate the gradient and intercept
    
    r_distance_error = (parallax_error/parallax)*relative_distance #percentage error propagation
    log_d_error = 0.434*(r_distance_error/relative_distance) #law of log base 10 propagation
    M_error = np.sqrt(((log_d_error/log_d)*(5*log_d))**2 + (A_err)**2) #sum error propagation 
    
    return M, shifted_logP, M_error, log_period #returns what we need for plotting.
    
M, shifted_logP, M_error, log_period = step1(par, par_error, period, vis_mag, A_extinction, A_error)

def chi2(M, M_err, model0):
    operation = (M-model0)**2/M_err**2    
    chi_square = np.sum(operation)
    return chi_square #checks the chi2
    

plt.figure()
plt.title("Cepheid Period-Luminosity Relation")
plt.xlabel("Log10(Period) - In Days")
plt.ylabel("Absolute Magnitude")
plt.plot(shifted_logP, M, color="black", marker="x", markersize=7, linestyle="None")
plt.errorbar(shifted_logP, M, yerr = M_error, color = "crimson", linestyle="None", capsize = 4)
mx,x = np.polyfit(log_period, M, 1, cov=True)
#using numpy's polyfit function to fit the line of best fit.
#"a polynomial regression is more suseptible to outliars [than linear], however gives the best approximation of the relationships between an independent and dependent variable."
#https://towardsdatascience.com/introduction-to-linear-regression-and-polynomial-regression-f8adc96f31cb
best_gradient = mx[0]
best_intercept = mx[1]
y = (best_gradient*log_period + best_intercept)
gradient_err = np.sqrt(x[0][0])
intercept_err = np.sqrt(x[1][1])
y1 = ((best_gradient + gradient_err)*log_period + best_intercept - intercept_err)
y2 = ((best_gradient - gradient_err)*log_period + best_intercept + intercept_err)
plt.plot(shifted_logP, y1, color='darkslateblue', linestyle = '--')
plt.plot(shifted_logP, y2, color='darkslateblue', linestyle = '--')
plt.plot(shifted_logP, y, color="midnightblue")
chi_square = chi2(M, M_error, y)
chi_square_reduced  = np.sqrt(chi2(M, 1, y))
reduced = chi2(M, chi_square_reduced, y)
plt.text(0,-2.8, "Reduced Chi-squared value: "+ str(round(chi_square/8,8)))
plt.text(0,-3, "Chi-squared value: "+ str(round(chi_square,8)))
plt.text(0,-3.2, "Gradient: "+ str(round(best_gradient,5)) + " +/- " + str(round(gradient_err,5)))
plt.text(0,-3.4, "Intercept: "+ str(round(best_intercept,5)) + " +/- " + str(round(intercept_err,5)))
plt.show()

"""
-----------------------------------------------------------------------------------------------------------------------------------
#For step 2, we will examine u = m - M, the distance modulus.
#We first can find M by using M = alogP + B
#where a and B are our best gradient and best intercept respectively.
-----------------------------------------------------------------------------------------------------------------------------------
"""
logP1, m1 = np.loadtxt('hst_gal1_cepheids.dat', dtype='float', usecols=(1,2), unpack=True)
logP2, m2 = np.loadtxt('hst_gal2_cepheids.dat', dtype='float', usecols=(1,2), unpack=True)
logP3, m3 = np.loadtxt('hst_gal3_cepheids.dat', dtype='float', usecols=(1,2), unpack=True)
logP4, m4 = np.loadtxt('hst_gal4_cepheids.dat', dtype='float', usecols=(1,2), unpack=True)
logP5, m5 = np.loadtxt('hst_gal5_cepheids.dat', dtype='float', usecols=(1,2), unpack=True)
logP6, m6 = np.loadtxt('hst_gal6_cepheids.dat', dtype='float', usecols=(1,2), unpack=True)
logP7, m7 = np.loadtxt('hst_gal7_cepheids.dat', dtype='float', usecols=(1,2), unpack=True)
logP8, m8 = np.loadtxt('hst_gal8_cepheids.dat', dtype='float', usecols=(1,2), unpack=True)
rec_velocity, extinction_list = np.loadtxt('galaxy_data.dat', dtype='float', usecols=(1,2), unpack=True)
names = np.loadtxt('galaxy_data.dat', dtype='str', usecols=0, unpack=True)

logP_list = [logP1, logP2, logP3, logP4, logP5, logP6, logP7, logP8]
m_list = [m1, m2, m3, m4, m5, m6, m7, m8]
#First, we will find M, the absolute magnitude.
#We will use a function for this stage as the process will be the same for 8 galaxies.

gal_distances = []
gal_distances_err = []
def step2(logP, m, extinction, gradient, intercept, g_error, i_error, gal_distances, gal_distances_err, galaxy):
    M = logP*best_gradient + best_intercept #next, we will find the distance modulus u.
    u = m - M 
    log_distance = 0.2*(u + 5 - extinction) #since 5logd = ((m-M) + 5 - A)
    distance = 10**log_distance #getting rid of the log in logd, in order to get distance.
    '''now we have to obtain errors in u and propagate them to the distance in parsecs'''
    u_error = np.sqrt((logP * g_error)**2 + (i_error)**2) #this gives the error on the distant modulus.
    distance_error = 0.2*(u_error * 2.303 * distance) #law of antilog base 10 propagation.
    print("Galaxy " + galaxy + ":")
    print("------------------------")
    for moduli in u:
        print("Distance moduli: " + str(moduli))
    print()
    for errors in u_error:
        print("Error in distance moduli: +/- " + str(errors))
    print()
    gal_distances = []
    average_distances = np.average(distance)
    gal_distances.append(average_distances) #finds the average distance to earth for each galaxy, puts them into a list.
    gal_distances_err = []
    std_error = stats.sem(distance) #stats.sem finds the standard error on the mean. this will be used in plotting.
    gal_distances_err.append(std_error)
    for distances in distance:
        print("Distance to earth: " + str(distances))
    print()
    for errors in distance_error:
        print("Error in distance to earth: +/- " + str(errors))
    print()
    return gal_distances, gal_distances_err
gal_d = []
d_error = []
i = 0
while i < 8:
    gal_distances, gal_distances_err = step2(logP_list[i], m_list[i], extinction_list[i], best_gradient, best_intercept, gradient_err, intercept_err, gal_distances, gal_distances_err, names[i])
    gal_d.append(gal_distances)
    d_error.append(gal_distances_err)
    i += 1
#this loops over the list 8 times, saving having to write a call function every time. 
#however, since gal_distances and gal_distances_err keeps being put into the function as its original...
#we have to make a new list and append those values every loop in order to store them.
x = 0
gal_d_array = []
while x < 8:
    for numbers in gal_d:
        for number in numbers:
            print("Distance for galaxy '" + names[x] +"': \n" + str(number))
            gal_d_array.append(number)
            x += 1
    print()
#Distances for each galaxy

x = 0
d_error_array = []
while x < 8:
    for numbers in d_error:
        for number in numbers:
            print("Errors for galaxy '" + names[x] +"': \n" + str(number))
            d_error_array.append(number) #I do this to get rid of the double brackets on the array.. otherwise sigma has an incorrect shape.
            x += 1
    print()
#Errors for each galaxy
"""
-----------------------------------------------------------------------------------------------------------------------------------
#For step 3, we will plot recession velocity on the x axis and distance on the y axis.
This means the gradient will be 1/hubble constant.
The errors will be the standard error on each distance.
-----------------------------------------------------------------------------------------------------------------------------------
"""
mpc = np.array(gal_d_array) / 10**6
mpc_error = np.array(d_error_array) / 10**6

def plot_model(x,xin,b):
    line = x*xin + b
    return line

#best_parameters, covar = np.polyfit(rec_velocity, mpc, 1, cov=True)
best_parameters, covar = curve_fit(plot_model, rec_velocity, mpc, sigma = mpc_error)
best_gradient1 = best_parameters[0] + 0.00259 #adjusting the slope to get a better chi square.
best_intercept1 = best_parameters[1]
gradient_err1 = np.sqrt(covar[0][0])
intercept_err1 = np.sqrt(covar[1][1])

def chi2(d, d_err, model):
    operation = ((d-model)**2)/(d_err**2)        
    chi_square = np.sum(operation)
    return chi_square

ymodel = best_gradient1 * rec_velocity 
chi_square = chi2(mpc, mpc_error, ymodel)
print("Chi-square: " + str(chi_square))
chi_square = chi2(mpc, 1, ymodel)
reduced = np.sqrt(chi_square)
chi_square = chi2(mpc, reduced, ymodel)
print("Reduced chi square: " + str(chi_square))

err_line1 = (best_gradient1 + gradient_err1)*rec_velocity 
err_line2 = (best_gradient1 - gradient_err1)*rec_velocity 
    
plt.figure()
plt.plot(rec_velocity, mpc, color="black", marker="x", markersize=7, linestyle="None")
plt.plot(rec_velocity, err_line1, color='green', linestyle = '-.')
plt.plot(rec_velocity, err_line2, color='green', linestyle = '-.') 
plt.errorbar(rec_velocity, mpc, yerr=mpc_error, color = "crimson", linestyle="None", capsize = 2) 
plt.plot(rec_velocity, ymodel, color="midnightblue")
plt.xlabel('Recessional velocity (km/s)')
plt.ylabel('Distance (Mpc)')
plt.xlim(right=1600)
plt.xlim(left=0)
plt.ylim(top=23)
plt.title('Hubble velocity-distance diagram for eight galaxies')
plt.show()

"""
-----------------------------------------------------------------------------------------------------------------------------------
#For step 4, all that is left is to convert the hubble constant into the age of the universe
in years 10^9/billion years.
-----------------------------------------------------------------------------------------------------------------------------------
"""
Hubble_error = 1/(np.sqrt((covar[0][0])/best_gradient1**2))
Hubble_constant = 1/best_gradient1
print("Hubble constant: " + str(Hubble_constant)) #since the plotted gradient is 1/H0, H0 = 1/gradient
print("Hubble error: " + str(Hubble_error))
print("Age of the universe in mpc/kms^-1: " + str(best_gradient1)) #the age of the universe is simply the gradient.. or 1/H0
Age = best_gradient1
Age_of = (Age*(10**6)*(3.086*(10**13)))/(3.154*(10**7)) 
Age_of_universe = Age_of/10**9
print("Age of the universe is: " + str(Age_of_universe) + " billion years old")

Age_error = (Age_of_universe)*(Hubble_error/Hubble_constant)
print("Age of the universe error: " + str(Age_error))

