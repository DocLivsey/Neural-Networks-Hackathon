import fs from 'fs';
import Papa from 'papaparse';
import {Recommender} from "./recommendation_algorithm.js";

export function readCSVFile(filePath, callback) {
    const fileContent = fs.readFileSync(filePath, 'utf8');

    Papa.parse(fileContent, {
        header: true,
        complete: function(results) {
            callback(results.data);
        },
        error: function(error) {
            console.error('Error parsing CSV:', error);
        }
    });
}

export function processUserData(data) {
    const user_data = {};

    data.forEach(row => {
        const user = row['review_profilename'];
        const beer_style = row['Styles'];
        const rating = parseFloat(row['review_overall']);
        if (!user_data[user]) {
            user_data[user] = {};
        }
        user_data[user][beer_style] = rating;
    });

    return user_data;
}

const filePath = './src/datasets/result.csv';

export function getResult(userRating, filePath) {
    let result = [];
    readCSVFile(filePath, (data) => {
        const user_data = processUserData(data);

        const r = new Recommender(user_data);
        r.computeDeviation();
        result = r.predictRating(userRating, 27)
    });
    return result;
}

console.log(getResult({'English Barleywine': 3, 'American Porter': 4}, filePath))
