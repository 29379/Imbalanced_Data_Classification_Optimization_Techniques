from read_data import read_data
from visualisation_methods import plot_in_2d


if __name__ == "__main__":
    X, y = read_data()
    plot_in_2d(X, y, "Classes")