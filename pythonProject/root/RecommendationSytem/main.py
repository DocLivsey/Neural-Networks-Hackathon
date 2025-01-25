from recommendation_algorithm import *


if __name__ == '__main__':
    file_path = './datasets/result.csv'
    data = read_csv(file_path)

    user_data = {}
    for index, row in data.iterrows():
        user = row['review_profilename']
        beer_style = row['Styles']
        rating = row['review_overall']
        user_data.setdefault(user, {})
        user_data[user][beer_style] = rating

    r = Recommender(user_data)
    r.computeDeviation()
    user = "DaPeculierDane" #Ник пользователя из датафрейма result.cvs
    u = user_data[user]
    print(r.predictRating(u, 27))
