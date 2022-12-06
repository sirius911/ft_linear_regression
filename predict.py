import yaml
import numpy as np
from Normalizer import Normalizer


green = '\033[92m' # vert
blue = '\033[94m' # blue
yellow = '\033[93m' # jaune
red = '\033[91m' # rouge
reset = '\033[0m' #gris, couleur normale

def load_model():
    print("Loading model ... ", end='')
    try:
        with open('model.yaml', 'r') as infile:
            model = yaml.safe_load(infile)
            print(f"{green}Ok{reset}")
            return model
    except Exception as e:
        print(f"{red}No Model available{reset}")
        return None

def control(mileage):
    try:
        x = int(mileage)
        return np.array([[mileage]], dtype=np.float64).reshape(-1, 1)
    except Exception:
        print(f"Please, enter an {blue}int{reset} !")
        return None

def main_loop():
    while 42:
        mileage = input("Tape the mileage in km (Nothing to quit): ")
        if mileage == '':
            return
        xx = control(mileage)
        if xx is not None:
            yy = theta[0] + theta[1] * xx
            price = round(yy[0][0], 2)
            if price < 0 :
                price = 0
            print(f"predicted price for {yellow}{mileage}{reset} km -> {green}{price:0.2f}{reset} $")


if __name__ == "__main__":
    print("*** Prediction ***")
    model = load_model()
    scaler_x = Normalizer()
    scaler_y = Normalizer()
    theta = (0.0, 0.0)

    if model is None:
        print("No trained model")
    else:
        theta = (model['theta0'], model['theta1'])
    print(f"Prediction with [{theta}]")
    main_loop()
    print("Good by !")