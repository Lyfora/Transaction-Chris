from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import os
import pickle
import joblib

app = Flask(__name__)

# Configuration
MODEL_FOLDER = 'models'
DATA_FOLDER = 'data'
KNN_MODEL_PATH = os.path.join(MODEL_FOLDER, 'knn_model.joblib')
CSV_DATA_PATH = os.path.join(DATA_FOLDER, 'transaction-engineered.csv')

# Helper function to create pivot table
def create_pivot_table(df):
    basket = df.pivot_table(
        index='Transaction_ID',
        columns='Deskripsi Barang',
        values='Jml',
        aggfunc='sum',
        fill_value=0
    )
    basket = (basket >= 1).astype(int)
    return basket
# Load id_barang mapping
id_barang = pd.read_csv(os.path.join(DATA_FOLDER, 'IdBarang.csv'))  # Assume it has columns ['id', 'product']

# Helper: Get product name from id
def get_product_by_id(product_id):
    try:
        return id_barang.iloc[product_id,1]
    except (IndexError, ValueError):
        return None
    
# Load data function
def load_data():
    df = pd.read_csv(CSV_DATA_PATH)
    return df

def train_apriori():
    # Get parameters from request
    min_support =  0.001
    min_threshold = 1
    
    # Load data directly
    df = load_data()
    basket = create_pivot_table(df)
    
    # Convert to binary (1 for purchased, 0 for not purchased)
    basket_binary = basket.applymap(lambda x: 1 if x > 0 else 0)
    basket_binary = basket_binary>0
    # Apply Apriori algorithm
    frequent_itemsets = apriori(basket_binary, min_support=min_support, use_colnames=True)
    
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=min_threshold)
    return rules

    
# Load model and Apriori Data
df = load_data()
basket = create_pivot_table(df)

model_knn = joblib.load(KNN_MODEL_PATH)
df_apriori = train_apriori()

# KNN recommendation endpoint (direct without training)
@app.route('/api/recommend/knn', methods=['GET'])
def recommend_knn():
    try:
        # Get data from query params
        product_id = int(request.args.get('id'))
        if not product_id:
            return jsonify({'error': 'Missing product ID'}), 400
            
        number_recommendation = int(request.args.get('number_recommendation', 5))
        try:
            number_recommendation = int(number_recommendation)
        except ValueError:
            return jsonify({'error': 'number_recommendation must be an integer'}), 400
        
        product = get_product_by_id(product_id)
        if not product or product not in basket.columns:
            return jsonify({'error': f'Product ID {product_id} not found'}), 404
        
        # Get recommendations
        product_index = list(basket.columns).index(product)
        distances, indices = model_knn.kneighbors(
            [basket.T.values[product_index]], 
            n_neighbors=min(number_recommendation+1, len(basket.columns))
        )
        
        recommendations = []
        for i in range(1, len(distances[0])):
            recommendations.append({
                'product': basket.columns[indices[0][i]],
                'similarity_score': float(1 - distances[0][i])
            })
        
        return jsonify({
            'product': product,
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# KNN Cross-Category recommendation endpoint
@app.route('/api/recommend/knn_cross', methods=['GET'])
def recommend_knn_cross():
    try:
        product_id = int(request.args.get('id'))
        if not product_id:
            return jsonify({'error': 'Missing product ID'}), 400
            
        number_recommendation_input = int(request.args.get('number_recommendation', 5))
        try:
            number_recommendation_input = int(number_recommendation_input)
        except ValueError:
            return jsonify({'error': 'number_recommendation must be an integer'}), 400
        
        number_recommendation = number_recommendation_input + 5

        product = get_product_by_id(product_id)
        if not product or product not in basket.columns:
            return jsonify({'error': f'Product ID {product_id} not found'}), 404
        
        # Verify that the product exists in the df DataFrame
        if product not in df['Deskripsi Barang'].values:
            return jsonify({'error': f'Product {product} not found in product database'}), 404
        
        product_index = list(basket.columns).index(product)
        target_jenis = df[df['Deskripsi Barang'] == product]['Jenis Alat'].values[0]
        
        distances, indices = model_knn.kneighbors(
            [basket.T.values[product_index]], 
            n_neighbors=min(number_recommendation+1, len(basket.columns))
        )
        
        recommendations = []
        count = 0
        for i in range(1, len(distances[0])):
            recommend_product = basket.columns[indices[0][i]]
            
            # Check if the recommended product exists in df DataFrame
            if recommend_product not in df['Deskripsi Barang'].values:
                continue
                
            recommend_jenis = df[df['Deskripsi Barang'] == recommend_product]['Jenis Alat'].values[0]
            
            if recommend_jenis != target_jenis:
                recommendations.append({
                    'product': recommend_product,
                    'similarity_score': float(1 - distances[0][i])
                })
                count += 1
            if count >= number_recommendation_input:
                break

        return jsonify({
            'product': product,
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Apriori List recommendation endpoint
@app.route('/api/recommend/apriori', methods=['GET'])
def recommend_apriori():
    try:
        product_id = int(request.args.get('id'))
        if not product_id:
            return jsonify({'error': 'Missing product ID'}), 400
            
        number_recommendation = int(request.args.get('number_recommendation', 5))
        try:
            number_recommendation = int(number_recommendation)
        except ValueError:
            return jsonify({'error': 'number_recommendation must be an integer'}), 400

        product = get_product_by_id(product_id)
        if not product:
            return jsonify({'error': f'Product ID {product_id} not found'}), 404
        
        all_antecedents = [list(x) for x in df_apriori['antecedents'].values]
        desired_indices = [i for i in range(len(all_antecedents)) if len(all_antecedents[i]) == 1 and all_antecedents[i][0] == product]

        if not desired_indices:
            return jsonify({'error': f"No association rules found for product '{product}'"}), 404

        apriori_recommendations = df_apriori.iloc[desired_indices].sort_values(by=['lift'], ascending=False)
        apriori_recommendations_list = [list(x) for x in apriori_recommendations['consequents'].values]

        recommendations = []
        for i in range(min(number_recommendation, len(apriori_recommendations_list))):
            recommendations.append({
                'product': apriori_recommendations_list[i][0] if len(apriori_recommendations_list[i]) == 1 else apriori_recommendations_list[i],
                'lift_score': float(apriori_recommendations.iloc[i]['lift'])
            })

        return jsonify({
            'product': product,
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Apriori Single-item recommendation endpoint
@app.route('/api/recommend/apriori_single', methods=['GET'])
def recommend_apriori_single():
    try:
        product_id = int(request.args.get('id'))
        if not product_id:
            return jsonify({'error': 'Missing product ID'}), 400
            
        number_recommendation = int(request.args.get('number_recommendation', 5))
        try:
            number_recommendation = int(number_recommendation)
        except ValueError:
            return jsonify({'error': 'number_recommendation must be an integer'}), 400

        product = get_product_by_id(product_id)
        if not product:
            return jsonify({'error': f'Product ID {product_id} not found'}), 404
        
        all_antecedents = [list(x) for x in df_apriori['antecedents'].values]
        desired_indices = [i for i in range(len(all_antecedents)) if len(all_antecedents[i]) == 1 and all_antecedents[i][0] == product]

        if not desired_indices:
            return jsonify({'error': f"No association rules found for product '{product}'"}), 404

        apriori_recommendations = df_apriori.iloc[desired_indices].sort_values(by=['lift'], ascending=False)
        apriori_recommendations_list = [list(x) for x in apriori_recommendations['consequents'].values]

        apriori_single_recommendations = apriori_recommendations.iloc[[i for i in range(len(apriori_recommendations_list)) if len(apriori_recommendations_list[i]) == 1]]
        apriori_single_recommendations_list = [list(x) for x in apriori_single_recommendations['consequents'].values]

        if not len(apriori_single_recommendations_list):
            return jsonify({'error': f"No single-item association rules found for product '{product}'"}), 404

        recommendations = []
        for i in range(min(number_recommendation, len(apriori_single_recommendations_list))):
            recommendations.append({
                'product': apriori_single_recommendations_list[i][0],  # Extract the item from the list
                'lift_score': float(apriori_single_recommendations.iloc[i]['lift'])
            })

        return jsonify({
            'product': product,
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500