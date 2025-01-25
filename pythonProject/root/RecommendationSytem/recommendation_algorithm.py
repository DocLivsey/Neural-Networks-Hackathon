import pandas as pd


class Recommender:

    def __init__(self, data):
        self.frequency = {}
        self.deviation = {}
        self.data = data

    # Рассчитываем отклонение оценок для всех пар сортов пива
    def computeDeviation(self):
        for user_ratings in self.data.values():
            for item, rating in user_ratings.items():
                self.frequency.setdefault(item, {})
                self.deviation.setdefault(item, {})
                for item2, rating2 in user_ratings.items():
                    if item != item2:
                        self.frequency[item].setdefault(item2, 0)
                        self.deviation[item].setdefault(item2, 0.0)
                        self.frequency[item][item2] += 1
                        self.deviation[item][item2] += (rating - rating2)
        for item, ratings in self.deviation.items():
            for item2 in ratings:
                ratings[item2] /= self.frequency[item][item2]

    # Прогноз оценки
    def predictRating(self, user_ratings, k):
        recommendations = {}
        frequencies = {}
        for item, rating in user_ratings.items():
            for diffItem, diffRating in self.deviation.items():
                if diffItem not in user_ratings and item in self.deviation[diffItem]:
                    fre = self.frequency[diffItem][item]
                    recommendations.setdefault(diffItem, 0.0)
                    frequencies.setdefault(diffItem, 0)
                    recommendations[diffItem] += (diffRating[item] + rating) * fre
                    frequencies[diffItem] += fre
        recommendations = [(k, v / frequencies[k]) for (k, v) in recommendations.items()]
        recommendations.sort(key=lambda a_tuple: a_tuple[1], reverse=True)
        return recommendations[:k]


def read_csv(file_path):
    return pd.read_csv(file_path)

