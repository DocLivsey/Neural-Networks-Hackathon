import { Recommender } from './recommendation_algorithm.js'

const data = {
    'user1': {'beer1': 5, 'beer2': 3, 'beer3': 2},
    'user2': {'beer1': 3, 'beer2': 4},
    'user3': {'beer2': 2, 'beer3': 5}
};

// Initialize and compute deviations
const rec = new Recommender(data);
rec.computeDeviation();

// Predict ratings for user3
const user_ratings = {'beer2': 2, 'beer3': 5};

const k = 1; // Number of recommendations
const recommendations = rec.predictRating(user_ratings, k);
console.log(recommendations);
