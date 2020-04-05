import numpy as np
from matplotlib import pyplot as plt

flag = 0

def pdf(points, mean_value, cov):
    cov=[cov]
    cov_inv = 1/cov[0]
    cov_inv = [cov_inv]
    diff = points - mean_value
    # print("diff=",diff.shape)

    diff_trans =np.transpose(diff)

    N = (2.0 * np.pi) ** (-1 / 2.0) * (1.0 / cov[0] ** 0.5) * \
        np.exp(-0.5 * np.multiply(np.multiply(diff_trans, cov_inv), diff))

    return N

def gaussian():

    mu1 = 0
    mu2 = 3
    mu3 = 6

    sigma1 = 2
    sigma2 = 0.5
    sigma3 = 3

    s1 = np.random.normal(mu1, sigma1, 100)
    s2 = np.random.normal(mu2, sigma2, 100)
    s3 = np.random.normal(mu3, sigma3, 100)
    X = np.array(list(s1) + list(s2) + list(s3))

    np.random.shuffle(X)

    bins = np.linspace(np.min(X), np.max(X), 100)

    plt.figure(figsize=(10, 7))
    plt.xlabel("$x$")
    plt.ylabel("pdf")
    plt.scatter(X, [0.005] * len(X), color='navy', s=30, marker=2, label="Train data")

    plt.title('Combined PDF and individual pdf ')
    # combined pdf func
    plt.plot(bins, pdf(bins, mu1, sigma1) + pdf(bins, mu2, sigma2) + pdf(bins, mu3, sigma3)  , color='red', lw = 10,label="True pdf")

    # individual pdf funs
    plt.plot(bins, pdf(bins, mu1, sigma1), color='green', label="Individual pdf plot")
    plt.plot(bins, pdf(bins, mu2, sigma2), color='blue', label="Individual pdf plot")
    plt.plot(bins, pdf(bins, mu3, sigma3), color='black', label="Individual pdf plot")
    plt.legend(loc='upper right')
    plt.savefig('Combined_EM.png')
    return bins,mu1,mu2,mu3,sigma1,sigma2,sigma3,X

def em():

    bins, mu1, mu2, mu3, sigma1, sigma2, sigma3, X = gaussian()
    k = 3
    weights = np.ones((k)) / k
    means = np.random.choice(X, k)
    variances = np.random.random_sample(size=k)
    # visualize the training data
    k=3
    z=0

    eps = 0.00001

    iterations = 1000
    iterations_completed = 0
    log_likelihood = []


    while iterations_completed<iterations:
        global flag
        z=z+1
        if flag==1:
            plt.figure(figsize=(10, 6))
            plt.xlabel("$x$")
            plt.ylabel("pdf")
            plt.title("Iteration {}".format(iterations_completed))
            plt.scatter(X, [0.005] * len(X), color='navy', s=30, marker=2, label="Train data")

            # combined pdf func
            plt.plot(bins,pdf(bins, means[0], variances[0]) + pdf(bins, means[1], variances[1]) + pdf(bins, means[2], variances[2]) , color='green', lw=8, label="True pdf")
            plt.plot(bins, pdf(bins, mu1, sigma1) + pdf(bins, mu2, sigma2) + pdf(bins, mu3, sigma3), color='red', lw=8,
                     label="True pdf")

            # individual funcs
            plt.plot(bins, pdf(bins, means[0], variances[0]), color='blue', label="Cluster 1")
            plt.plot(bins, pdf(bins, means[1], variances[1]), color='pink', label="Cluster 2")
            plt.plot(bins, pdf(bins, means[2], variances[2]), color='magenta', label="Cluster 3")

            plt.legend(loc='upper left')

            plt.show()
    # calculate the maximum likelihood of each observation xi
        likelihood_list = []
        log_likelihood_list = []

        # Expectation step
        for j in range(k):
            likelihood_list.append(pdf(X, means[j], np.sqrt(variances[j])))
            log_likelihood_list.append(pdf(X, means[j], np.sqrt(variances[j]))*weights[j])

        likelihood_list = np.array(likelihood_list)
        log_likelihood_list = np.array(log_likelihood_list).T
        likelihood_sum = np.sum(log_likelihood_list,axis=1)
        log_value =  np.sum(np.log(likelihood_sum))
        log_likelihood.append(log_value)

        b = []

        # Maximization step
        for j in range(k):
            # use the current values for the parameters to evaluate the posterior
            # probabilities of the data to have been generanted by each gaussian
            b.append((likelihood_list[j] * weights[j]) / (np.sum([likelihood_list[i] * weights[i] for i in range(k)], axis=0) + eps))

            # updage mean and variance
            means[j] = np.sum(b[j] * X) / (np.sum(b[j]+eps))
            variances[j] = np.sum(b[j] * np.square(X - means[j])) / (np.sum(b[j] + eps))

            # update the weights
            weights[j] = np.mean(b[j])

        if len(log_likelihood) < 2:
            continue
        if np.abs(log_value - log_likelihood[-2]) < eps:
            flag = 1
            plt.figure(figsize=(10, 6))
            plt.xlabel("$x$")
            plt.ylabel("pdf")
            plt.title(" Best result reached at Iteration {}".format(iterations_completed))
            plt.scatter(X, [0.005] * len(X), color='navy', s=30, marker=2, label="Randomly generated data points")


            plt.plot(bins, pdf(bins, mu1, sigma1), color='red', label="Cluster 1")
            plt.plot(bins, pdf(bins, mu2, sigma2), color='green', label="Cluster 2")
            plt.plot(bins, pdf(bins, mu3, sigma3), color='brown', label="Cluster 3")

            # individual funcs
            plt.plot(bins, pdf(bins, means[0], variances[0]), color='blue', label="Cluster 1 after GMM training")
            plt.plot(bins, pdf(bins, means[1], variances[1]), color='pink', label="Cluster 2 after GMM training")
            plt.plot(bins, pdf(bins, means[2], variances[2]), color='magenta', label="Cluster 3 after GMM training")

            plt.legend(loc='upper left')
            plt.savefig('Individual_EM.png')
            plt.show()
            break
        iterations_completed+=1
em()