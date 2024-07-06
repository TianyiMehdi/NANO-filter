import numpy as np
from filterpy.monte_carlo import systematic_resample
from numpy.random import uniform
from scipy.stats import multivariate_normal,norm


class PF:
    def __init__(self, model, num_particles=100):

        self.f = model.f
        self.h = model.h
        self.dim_x=model.dim_x
        self.dim_y=model.dim_y

        self.Q = model.Q
        self.R = model.R
        self.x = model.x0
        self.P = model.P0
        self.ex = np.zeros(self.dim_x)
        self.ey = np.zeros(self.dim_y)

        self.num_particles = num_particles
        self.particles = np.random.multivariate_normal(mean=self.x, cov=self.P,
                                                       size=self.num_particles)
        self.weights = np.ones(num_particles) / num_particles
        self.x_prior = self.x.copy()
        self.x_post = self.x.copy()

    def predict(self):
        for i in range(self.num_particles):
            self.particles[i] = self.f(self.particles[i]) + np.random.multivariate_normal(self.ex,
                                                                                          self.Q)
        self.x_prior = np.average(self.particles, weights=self.weights, axis=0)
        self.x = self.x_prior.copy()

    def update(self, y):

        for i in range(self.num_particles):
            y_pred = self.h(self.particles[i])
            self.weights[i] *= multivariate_normal.pdf(y, y_pred, self.R)

        self.weights += 1.e-300
        self.weights /= sum(self.weights)

        # 2. 重采样
        if self.neff(self.weights) < self.num_particles / 2:
            indexes = systematic_resample(self.weights)
            self.resample(indexes)
        self.x_post = np.average(self.particles, weights=self.weights, axis=0)
        self.x = self.x_post.copy()

    def neff(self, weights):
        return 1. / np.sum(np.square(weights))

    def resample(self, indexes):
        self.particles[:] = self.particles[indexes]
        self.weights.resize(len(self.particles))
        self.weights.fill(1.0 / len(self.weights))

