import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

class RegressionVisualizer:
    def __init__(self, x_min, x_max, n_points, noise_mean, noise_sd, ridge_alpha, degree_max):
        self.x_min = x_min
        self.x_max = x_max
        self.n_points = n_points
        self.noise_mean = noise_mean
        self.noise_sd = noise_sd
        self.ridge_alpha = ridge_alpha
        self.degree_max = degree_max
        self.x_smooth = np.linspace(x_min, x_max, 1001)
        self.X = np.linspace(x_min, x_max, n_points)
        self.X_sample = x_min + np.random.rand(n_points) * (x_max - x_min)
        self.y = self._generate_noisy_data()
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X.reshape(-1, 1))
    
    def _generate_noisy_data(self):
        noise = np.random.normal(loc=self.noise_mean, scale=self.noise_sd, size=self.n_points)
        return self.func(self.X) + noise
    
    @staticmethod
    def func(x):
        return x**2 * np.sin(x) * np.exp(-(1 / 10) * x)
    
    def train_linear_regression(self):
        linear_reg = LinearRegression()
        linear_reg.fit(self.X_scaled, self.y)
        return linear_reg.predict(self.X_scaled)
    
    def train_polynomial_ridge_regression(self, alpha, degree):
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(self.X_scaled)
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_poly, self.y)
        return ridge.predict(X_poly)
    
    def plot_linear_regression(self, save_path):
        y_linear_pred = self.train_linear_regression()
        plt.figure(figsize=(10, 5))
        plt.scatter(self.X, self.y, color='orange', label='Noisy samples')
        plt.plot(self.x_smooth, self.func(self.x_smooth), 'k', label='Ideal function')
        plt.plot(self.X, y_linear_pred, color='blue', label='Linear Regression')
        plt.title("Linear Regression")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.show()
    
    def plot_polynomial_ridge(self, alpha, degree, save_path):
        y_poly_pred = self.train_polynomial_ridge_regression(alpha, degree)
        plt.figure(figsize=(10, 5))
        plt.scatter(self.X, self.y, color='orange', label='Noisy samples')
        plt.plot(self.x_smooth, self.func(self.x_smooth), 'k', label='Ideal function')
        plt.plot(self.X, y_poly_pred, color='red', label=f'Polynomial Ridge (α={alpha})')
        plt.title(f"Polynomial Ridge Regression (Degree={degree}, α={alpha})")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.show()
    
    def plot_comparison(self, alpha, degree, save_path):
        y_linear_pred = self.train_linear_regression()
        y_poly_pred = self.train_polynomial_ridge_regression(alpha, degree)
        plt.figure(figsize=(10, 5))
        plt.scatter(self.X, self.y, color='orange', label='Noisy samples')
        plt.plot(self.x_smooth, self.func(self.x_smooth), 'k', label='Ideal function')
        plt.plot(self.X, y_linear_pred, color='blue', label='Linear Regression')
        plt.plot(self.X, y_poly_pred, color='red', label=f'Polynomial Ridge (α={alpha})')
        plt.title(f"Linear vs Polynomial Ridge Regression (Degree={degree}, α={alpha})")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.show()


x_min = 1
x_max = 10
n_points = 41
noise_mean = 0
noise_sd = 2
ridge_alpha = tuple([10**(x) for x in range(-3, 0, 1)])
degree_max = 8

visualizer = RegressionVisualizer(x_min, x_max, n_points, noise_mean, noise_sd, ridge_alpha, degree_max)
    
visualizer.plot_linear_regression("linear_regression.png")
    
for alpha in ridge_alpha:
    visualizer.plot_polynomial_ridge(alpha, degree_max, f"polynomial_ridge_alpha_{alpha}.png")
    visualizer.plot_comparison(alpha, degree_max, f"comparison_alpha_{alpha}.png")
