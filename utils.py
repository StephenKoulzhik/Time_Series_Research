import numpy as np
from scipy.special import gamma
from scipy.spatial.distance import pdist, squareform, euclidean
from tqdm import trange, tqdm
from collections import defaultdict
from itertools import combinations, product, groupby


class Lorentz:
    def __init__(self, s=10.0, r=28.0, b=8 / 3):
        self.s = s
        self.r = r
        self.b = b

    # Differential equations of a Lorenz System
    def X(self, x, y, s):
        return s * (y - x)

    def Y(self, x, y, z, r):
        return (-x) * z + r * x - y

    def Z(self, x, y, z, b):
        return x * y - b * z

    # RK4 for the differential equations
    def RK4(self, x, y, z, s, r, b, dt):
        k_1 = self.X(x, y, s)
        l_1 = self.Y(x, y, z, r)
        m_1 = self.Z(x, y, z, b)

        k_2 = self.X((x + k_1 * dt * 0.5), (y + l_1 * dt * 0.5), s)

        l_2 = self.Y(
            (x + k_1 * dt * 0.5), (y + l_1 * dt * 0.5), (z + m_1 * dt * 0.5), r
        )

        m_2 = self.Z(
            (x + k_1 * dt * 0.5), (y + l_1 * dt * 0.5), (z + m_1 * dt * 0.5), b
        )

        k_3 = self.X((x + k_2 * dt * 0.5), (y + l_2 * dt * 0.5), s)

        l_3 = self.Y(
            (x + k_2 * dt * 0.5), (y + l_2 * dt * 0.5), (z + m_2 * dt * 0.5), r
        )

        m_3 = self.Z(
            (x + k_2 * dt * 0.5), (y + l_2 * dt * 0.5), (z + m_2 * dt * 0.5), b
        )

        k_4 = self.X((x + k_3 * dt), (y + l_3 * dt), s)
        l_4 = self.Y((x + k_3 * dt), (y + l_3 * dt), (z + m_3 * dt), r)
        m_4 = self.Z((x + k_3 * dt), (y + l_3 * dt), (z + m_3 * dt), b)

        x += (k_1 + 2 * k_2 + 2 * k_3 + k_4) * dt * (1 / 6)
        y += (l_1 + 2 * l_2 + 2 * l_3 + l_4) * dt * (1 / 6)
        z += (m_1 + 2 * m_2 + 2 * m_3 + m_4) * dt * (1 / 6)

        return (x, y, z)

    def generate(self, dt, steps):

        # Initial values and Parameters
        x_0, y_0, z_0 = 1, 1, 1

        # RK4 iteration
        x_list = [x_0]
        y_list = [y_0]
        z_list = [z_0]

        i = 0

        while i < steps:
            x = x_list[i]
            y = y_list[i]
            z = z_list[i]

            position = self.RK4(x, y, z, self.s, self.r, self.b, dt)

            x_list.append(position[0])
            y_list.append(position[1])
            z_list.append(position[2])

            i += 1

        x_array = np.array(x_list)
        y_array = np.array(y_list)
        z_array = np.array(z_list)

        return x_array, y_array, z_array


def partition(dist, l, r, order):
    if l == r:
        return l

    pivot = dist[order[(l + r) // 2]]
    left, right = l - 1, r + 1
    while True:
        while True:
            left += 1
            if dist[order[left]] >= pivot:
                break

        while True:
            right -= 1
            if dist[order[right]] <= pivot:
                break

        if left >= right:
            return right

        order[left], order[right] = order[right], order[left]


def nth_element(dist, order, k):
    l, r = 0, len(order) - 1
    while True:
        if l == r:
            break
        m = partition(dist, l, r, order)
        if m < k:
            l = m + 1
        elif m >= k:
            r = m


def volume(r, m):
    return np.pi ** (m / 2) * r**m / gamma(m / 2 + 1)


def significant(cluster, h, p):
    max_diff = max(abs(p[i] - p[j]) for i, j in product(cluster, cluster))

    # print(max_diff)
    return max_diff >= h


def get_clustering(x, k, h, verbose=True):
    n = len(x)
    if isinstance(x[0], list):
        m = len(x[0])
    else:
        m = 1
    dist = squareform(pdist(x))

    dk = []
    for i in range(n):
        order = list(range(n))
        nth_element(dist[i], order, k - 1)
        dk.append(dist[i][order[k - 1]])

    # print(dk)

    p = [k / (volume(dk[i], m) * n) for i in range(n)]

    w = np.full(n, 0)
    completed = {0: False}
    last = 1
    vertices = set()
    for d, i in sorted(zip(dk, range(n))):
        neigh = set()
        neigh_w = set()
        clusters = defaultdict(list)
        for j in vertices:
            if dist[i][j] <= dk[i]:
                neigh.add(j)
                neigh_w.add(w[j])
                clusters[w[j]].append(j)

        vertices.add(i)
        if len(neigh) == 0:
            w[i] = last
            completed[last] = False
            last += 1
        elif len(neigh_w) == 1:
            wj = next(iter(neigh_w))
            if completed[wj]:
                w[i] = 0
            else:
                w[i] = wj
        else:
            if all(completed[wj] for wj in neigh_w):
                w[i] = 0
                continue
            significant_clusters = set(
                wj for wj in neigh_w if significant(clusters[wj], h, p)
            )
            if len(significant_clusters) > 1:
                w[i] = 0
                for wj in neigh_w:
                    if wj in significant_clusters:
                        completed[wj] = wj != 0
                    else:
                        for j in clusters[wj]:
                            w[j] = 0
            else:
                if len(significant_clusters) == 0:
                    s = next(iter(neigh_w))
                else:
                    s = next(iter(significant_clusters))
                w[i] = s
                for wj in neigh_w:
                    for j in clusters[wj]:
                        w[j] = s
    return w


def generate_centers(x_trains, N, WISHART_K=4, WISHART_H=0.2):
    ws = {}
    for pattern, train in x_trains.items():
        ws[pattern] = get_clustering(train, WISHART_K, WISHART_H)

    centers = {}

    for pattern, w in ws.items():
        sorted_by_cluster = sorted(range(len(w)), key=lambda x: w[x])
        for wi, cluster in groupby(sorted_by_cluster, lambda x: w[x]):
            cluster = list(cluster)
            center = np.full(N, 0.0)
            for i in cluster:

                center += x_trains[pattern][i]
            centers.setdefault(pattern, []).append(center / len(cluster))

    return centers


class SimpleDeamon(object):
    def __init__(self, mode="simple"):
        self.mode = mode
        self.predictions = {
            point: [None for i in range(point + 1)] for point in range(PTS)
        }
        self.set_predictions = {
            point: [None for i in range(point + 1)] for point in range(PTS)
        }

        self.predicted = False

    @property
    def label(self):
        return "Simple model of demon with " + self.mode + " mode"

    def simple_aggr(pts):
        if not pts:
            return None
        return np.mean(np.array(list(map(lambda center: center[0], pts))))

    def predict(self, start_point, step, preds):
        self.set_predictions[start_point].append(preds)

        if self.mode == "simple":
            pred = simple_aggr(preds)
        elif self.mode == "d_weighted":
            pred = aggr_d(preds)
        elif self.mode == "q_weighted":
            pred = aggr_q(preds)
        elif self.mode == "mix":
            pred = aggr_mix(preds)

        self.predictions[start_point].append(pred)
        return pred

    def get_predictions(self):
        return self.predictions

    def get_set_predictions(self):
        return self.set_predictions

    def is_predicted(self):
        return self.predicted


def generate_predictions(
    centers,
    deamon=None,
    return_set_pred=False,
    real_mode="test",
    EPS=0.05,
    Q_VALUE=0.99,
    PTS=100,
    STEPS=50,
):
    preds = {}
    set_preds = {}

    if real_mode == "test":
        end_point = val_end
        init_point = test_init
    else:
        end_point = train_end
        init_point = val_init

    for start_point in tqdm(range(PTS)):
        # initialize empty
        preds[start_point] = [None] * (start_point + 1)
        if return_set_pred:
            set_preds[start_point] = [None] * (start_point + 1)

        # current window
        wind = list(
            map(
                lambda x: (x, 1), xs[end_point + start_point : init_point + start_point]
            )
        )

        for step in range(1, STEPS + 1):
            x_tests_for_point = {}
            for pattern in patterns:

                key = str_subseq(pattern + (WINDOW - 1,))
                sample = gen_sample_in_point_with_q(
                    np.concatenate([wind, [(0, 0)]]), WINDOW, pattern, len(wind)
                )
                if not sample:
                    x_tests_for_point[key] = None
                else:
                    x_tests_for_point[key] = sample

            chosen_centers = []
            for pattern, centers_values in centers.items():
                if not x_tests_for_point[pattern]:
                    continue
                vector = np.array(x_tests_for_point[pattern][:-1])[:, 0]
                q_values = np.array(x_tests_for_point[pattern][:-1])[:, 1]

                for center in centers_values:
                    dist = euclidean(vector, center[:-1])
                    if dist < EPS:
                        weight_d = (EPS - dist) / EPS
                        weight_q = np.mean(q_values) * Q_VALUE
                        chosen_centers.append((pattern, center, weight_d, weight_q))

            last_points = list(
                map(
                    lambda center: (center[1][-1], center[2], center[3], center[0]),
                    chosen_centers,
                )
            )

            # deamon predict
            result_point = deamon.predict(start_point, step, last_points)
            preds[start_point].append(result_point)

            if return_set_pred:
                set_preds[start_point].append(last_points)

            if result_point:
                q_value = np.mean(
                    np.array(list(map(lambda center: center[2], chosen_centers)))
                )
            else:
                q_value = None

            # move the window
            wind = np.concatenate([wind[1:], [(result_point, q_value)]])

    deamon.predicted = True

    if return_set_pred:
        return preds, set_preds
    return preds
