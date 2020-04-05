import numpy as np
from matplotlib import pyplot as plt
import glob
import imageio

# Generate data
def image_points():
    image_pixels = []
    # read the trained images from the train folder
    for image_path in glob.glob("buoy_Orange\\Train\\*.png"):
        img = imageio.imread(image_path)
        orange_values = img[:, :, 0]
        # read pixel intensities from red channel
        r, c = orange_values.shape
        for j in range(0, r):
            for m in range(0, c):
                im = orange_values[j][m]
                image_pixels.append(im)
    image_pixels = np.array(image_pixels)
    # reshape it to 1-D array
    data = np.reshape(image_pixels, (len(image_pixels), 1))
    return data


# In[3]:
data = image_points()
K = 3
# In[4]:
def gaussian_pdf(data, mean, covar):
    data_mean = np.matrix(data - mean)
    covar_inv = np.linalg.pinv(covar)
    pdf = (2.0 * np.pi) ** (-len(data[1]) / 2.0) * (1.0 / np.linalg.det(covar) ** 0.5) * \
          np.exp(-0.5 * np.sum(np.multiply(data_mean * covar_inv, data_mean), axis=1))
    return pdf


# generate initial parameters
def initial_parameters(data, K):
    row, col = data.shape
    # initialize mean values reandomly from the distribution
    mean = np.array(data[np.random.choice(row, K)], np.float64)
    # initialize an identity matrix
    covar = [np.random.randint(1, 255) * np.eye(col)] * K
    for sig in range(K):
        covar[sig] = np.multiply(covar[sig], np.random.rand(col, col))
    # initialize prior probablities equally such that sum is 1
    weights = [1. / K] * K
    return mean, covar, weights


mean,covar,weights = initial_parameters(data,K)

print("initial mean=",mean)
print("initial covar=",covar)
print("initial weights=",weights)


# In[5]:


# Training function - updation of Parameters
def train_GMM(data, K, mean, covar, weights):
    row, col = data.shape
    likelihood_prob = np.zeros((row, K))
    # create a list to store log_likelihood values
    log_likelihood = 0
    log_likelihood_values = []
    iterations_completed = 0
    # define a limit to max no if iterations
    max_iterations = 1000
    while (iterations_completed < max_iterations):
        prev_log = log_likelihood
        iterations_completed += 1
        ######## Expectaion Step##########
        # For summation, axis = 1 -> sum along clusters
        #  axis = 0 -> sum along the data points
        for i in range(K):
            # calulate the likelihood probablity (Numerator in the formula)
            likelihood_prob[:, i:i + 1] = gaussian_pdf(data, mean[i], covar[i]) * weights[i]
            # likelihood_prob[:,i] = np.array(likelihood_prob_a)
        # caclulate the log likelihood by summating the likelihood probabilities for Maximum likelihood Estimation
        log_likelihood = np.sum(np.log(np.sum(likelihood_prob, axis=1)))
        log_likelihood_values.append(log_likelihood)
        # calculate the sum of the likelihood probabilities along all the clusters (Denominator in the formula)
        evidence = np.sum(likelihood_prob, axis=1)
        # Divide the numerator and denominator for posterior ie. given the points calculate the prob that the points belong to cluster Cj
        posterior_T = likelihood_prob.T / evidence
        posterior = posterior_T.T
        # calculate the posterior sum for mean updation
        posterior_sum = np.sum(posterior, axis=0)

        ######### Maximazation Step###########
        for j in range(K):
            # update values
            # sum of posterior probablities x data / sum of posterior probablities
            mean[j] = 1. / posterior_sum[j] * np.sum(posterior[:, j] * data.T, axis=1).T
            x_mean = data - mean[j]
            # covariance matrox - (sum of posterior * (data-mean).T* (data-mean))/sum of posterior
            covar[j] = np.array(1. / posterior_sum[j] * np.dot(np.multiply(x_mean.T, posterior[:, j]), x_mean))
            # sum of posterior/ no. of data points
            weights[j] = (1. / row) * posterior_sum[j]

        if np.abs(log_likelihood - prev_log) < 0.0001:
            print("Converged")
            print("iterations_completed=", iterations_completed)
            # plotting the likelihood graph
            plt.plot(log_likelihood_values)
            plt.title("Log-likelihood vs Iterations -  RED BUOY")
            plt.savefig('red_log_likelihood.png')
            plt.show()
            return mean, covar, weights
            break


#calculate the optimal cluster paramters
updated_mean,updated_covar,updated_weights = train_GMM(data,K,mean,covar,weights)
np.save('red_means.npy',updated_mean)
np.save('red_covar.npy',updated_covar)
np.save('red_weights.npy',updated_weights)
print("updated parameters=",updated_mean,updated_covar,updated_weights)



