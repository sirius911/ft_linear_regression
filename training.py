import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml

from mylinearregression import MyLinearRegression as MyLR
from Normalizer import Normalizer


green = '\033[92m' # vert
blue = '\033[94m' # blue
yellow = '\033[93m' # jaune
red = '\033[91m' # rouge
reset = '\033[0m' #gris, couleur normale


def main():
    print("Loading data ...")
    try:
    # Importation of the dataset
        data = pd.read_csv("data.csv", dtype=np.float64)
    except:
        print("Issue when trying to retrieve the dataset.", file=sys.stderr)
        sys.exit()
    km = data.km.values.reshape(-1, 1) #km
    price = data.price.values.reshape(-1, 1) # price 

    # normalizer
    print("Normalizer ...", end='')
    scaler_x = Normalizer(km)
    scaler_y = Normalizer(price)
    x = scaler_x.norme(km)
    y = scaler_y.norme(price)
    print(" ok")

    #training
    print("training")
    thetas = np.array([[0.0, 0.0]]).reshape(-1, 1)
    iter = 10000
    alpha = 1e-3

    mylr = MyLR(thetas, alpha, iter, progress_bar=True)

    evol_mse = mylr.fit_(x, y)
    y_hat = mylr.predict_(x)
    mse = MyLR.mse_(y, y_hat)
    print(mylr.thetas)

    #graph
    plt.figure()
    plt.plot(np.arange(iter),evol_mse)
    plt.xlabel("iterations")
    label = f"\nMse = {mse}"
    plt.ylabel("mse")
    plt.title(rf'$\theta_0 = {float(mylr.thetas[0][0]):0.2e}~&~\theta_1 = {float(mylr.thetas[1][0]):0.2e}${label}')

    plt.figure()
    plt.scatter(km, price, c="b", marker='o', label='real data')
    plt.xlabel("km")
    plt.ylabel("price")
    y_hat = mylr.predict_(x)
    y_hat = scaler_y.inverse(y_hat)
    plt.plot(km, y_hat, c='r', label='predicted price')
    plt.legend()
    plt.show()

    #save model
    print(f"Saving model with theta0 = {yellow}{float(mylr.thetas[0][0]):0.2e}{reset} and thetha1 = {yellow}{float(mylr.thetas[1][0]):0.2e}{reset}... ", end='')
    model = {}
    model['theta0'] = float(mylr.thetas[0][0])
    model['theta1'] = float(mylr.thetas[1][0])
    model['mean_x'] = float(scaler_x.mean_)
    model['mean_y'] = float(scaler_y.mean_)
    model['std_x'] = float(scaler_x.std_)
    model['std_y'] = float(scaler_y.std_)
    print(f"{green}Ok{reset}")

    with open('model.yaml', 'w') as outfile:
        yaml.dump(model, outfile, sort_keys=False, default_flow_style=None )

if __name__ == "__main__":
    print("trainig starting ...")
    main()
    print("Good by !")