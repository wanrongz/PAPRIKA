import numpy as np
from toimport import *

from scipy.stats import norm
from scipy.stats import truncexpon
from scipy.stats import bernoulli
from scipy.special import comb 

class rowexp_new_batch:

    def __init__(self, NUMHYP, numdraws, alt_vec, mu0, mu_alt_vec, pi):
        self.numhyp = NUMHYP
        self.alt_vec = alt_vec
        self.mu0 = mu0
        self.mu_vec = mu0*np.ones(NUMHYP) + np.multiply(alt_vec, mu_alt_vec)
        self.pvec = np.zeros(NUMHYP)
        self.numdraws = numdraws
        self.pi=pi

        '''
        Function drawing p-values: Mixture of two Gaussians
        '''
    def gauss_two_mix(self, mu_gap, sigma = 1, rndsd = 0):

        np.random.seed(rndsd)

        # Draw Z values according to lag
        Z = self.mu_vec + np.random.randn(self.numhyp)*sigma # draw gaussian acc. to hypothesis, if sigma are all same


        # Compute p-values and save
        if mu_gap > 0:
            self.pvec = [(1 - norm.cdf(z)) for z in Z] # one-sided p-values
        else:
            self.pvec = [2*norm.cdf(-abs(z)) for z in Z] # two-sided p-values

    def beta_draws(self, rndsd = 0):
        np.random.seed(rndsd)
        self.pvec = [(np.random.beta(0.5,5,1)*self.alt_vec[i]+np.random.uniform(0,1,1)*(1-self.alt_vec[i])) for i in range(self.numhyp)]

    def bernoulli_draws(self, theta1, theta2, rndsd=1, datasize=1000):
        np.random.seed(rndsd)
        for i in range(self.numhyp):
            if self.alt_vec[i]==0:
                database=bernoulli.rvs(theta1,size=datasize)
            else:
                database=bernoulli.rvs(theta2,size=datasize)
            t=sum(database)
            pval=0
            for j in range(t,datasize+1):
                pval=pval+1/(2**datasize)*comb(datasize,j,exact=True)
            self.pvec[i]=pval 
        dirname = './expsettings'
        filename = "P_NH%d_PM%.2f_T1%.2f_T2%.2f" % (self.numhyp, self.pi, theta1, theta2)
        saveres(dirname, filename, self.pvec)

        
    def get_bernoulli_draws(self, rndsd=1, theta1=0.5, theta2=0.75, datasize=1000):
        
        # Read pvalues from file
        filename_pre = "P_NH%d_PM%.2f_T1%.2f_T2%.2f" % (self.numhyp, self.pi, theta1, theta2)
        p_filename = [filename for filename in os.listdir('./expsettings') if filename.startswith(filename_pre)]
        if len(p_filename) > 0:
        # Just take the first sample
            self.pvec = np.loadtxt('./expsettings/%s' % p_filename[0])    
        else:
            #print("Hyp file doesn't exist, thus generating the file now ...")
        # Generate pvec with given setting
            self.bernoulli_draws(theta1, theta2, rndsd, datasize=1000)
    
    def truncexpon_draws(self, lbd_scale, rndsd=0, thresh=1, datasize=1000):
        np.random.seed(rndsd)
        for i in range(self.numhyp):
            if self.alt_vec[i]==0:
                database=truncexpon.rvs(b=thresh, size=datasize)
            else:
                database=truncexpon.rvs(b=thresh, scale=lbd_scale, size=datasize)
            z=sum(database)
            pval=1 - norm.cdf(z,loc=datasize*(1+1/(1-np.exp(1))), scale=datasize*(1-np.exp(1)/(np.exp(1)-1)**2))
            self.pvec[i]=pval 
        dirname = './expsettings'
        filename = "P_NH%d_PM%.2f_lbd%.2f_SEED%d" % (self.numhyp, self.pi, lbd_scale,rndsd)
        saveres(dirname, filename, self.pvec)

        
    def get_truncexpon_draws(self, rndsd=0, lbd_scale=2.00, datasize=1000):
        
        # Read pvalues from file
        filename_pre = "P_NH%d_PM%.2f_lbd%.2f_SEED%d" % (self.numhyp, self.pi, lbd_scale,rndsd)
        p_filename = [filename for filename in os.listdir('./expsettings') if filename.startswith(filename_pre)]
        if len(p_filename) > 0:
        # Just take the first sample
            self.pvec = np.loadtxt('./expsettings/%s' % p_filename[0])    
        else:
            #print("Hyp file doesn't exist, thus generating the file now ...")
        # Generate pvec with given setting
            self.truncexpon_draws(lbd_scale, rndsd, thresh=1, datasize=1000)
