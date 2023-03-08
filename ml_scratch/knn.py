from collections import Counter

from ml_scratch.metrics.distance_metrics import(
    euclidean_distance,
    minkowski_distance
)


class KNNClassifier():
    def __init__(self, n_neighbors=3, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric
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
        y_pred = []

        for test_coord in X_test:
            prediction = self._predict(test_coord)
            y_pred.append(prediction)

        return y_pred
    
    def _predict(self, test_coord):
        distances = []

        for train_coord, target in zip(self.X_train, self.y_train):
            distance = self.metrics.get(self.metric)(test_coord, train_coord)
            distances.append((distance, target))

            neighbors = sorted(distances)[:self.n_neighbors]
            neighbors_target = [i[1] for i in neighbors]
            target_counter = Counter(neighbors_target)
            prediction = target_counter.most_common()[0][0]

        return prediction
