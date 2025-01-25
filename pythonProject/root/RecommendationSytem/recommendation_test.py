from recommendation_algorithm import Recommender

data = {
    'user1': {'beer1': 5, 'beer2': 3, 'beer3': 2},
    'user2': {'beer1': 3, 'beer2': 4},
    'user3': {'beer2': 2, 'beer3': 5}
}

# Инициализация и расчет отклонений
rec = Recommender(data)
rec.computeDeviation()

# Прогнозирование оценок для user2
#user_ratings = {'beer1': 3, 'beer2': 4}

#Прогнозирование оценки для user3
user_ratings = {'beer2': 2, 'beer3': 5}

k = 1  # Количество рекомендаций
recommendations = rec.predictRating(user_ratings, k)
print(recommendations)
