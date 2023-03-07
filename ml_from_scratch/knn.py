from metrics.distance_metrics import(
    euclidean_distance,
    minkowski_distance
)

class KNeighborsClassifier():
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.metric = 'euclidean'
        self.X_train = None
        self.y_train = None

        self.metrics = {
            'euclidean': euclidean_distance,
            'minkowski': minkowski_distance
        }

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        for test_coord in X_test:
            distances = []
    
            for train_coord, target in zip(self.X_train, self.y_train):
                distance = self.metrics.get(
                    self.metric
                )(test_coord, train_coord)

                distances.append((distance, target))

if __name__ == '__main__':
    print(euclidean_distance((2,2,2,2), (4,4,4,4)))