# recommender.py

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import os

# --- Global Data Structures (Set up once) ---
df = None
transactions = None
df_encoded = None
RULES_DF = None

# --- Helper Function for Ingredient Cleaning (from preprocess.py) ---
def clean_ingredients(ingredient_list):
    """Normalizes ingredient names for both dataset and user input."""
    synonym_map = {
        'mustard oil': 'oil', 'vegetable oil': 'oil', 'sunflower oil': 'oil', 'sesame oil': 'oil', 
        'peanut oil': 'oil', 'olive oil': 'oil',
        'clarified butter': 'ghee', 'butter': 'ghee',
        'maida flour': 'maida / all-purpose flour', 'plain flour': 'maida / all-purpose flour', 
        'white flour': 'maida / all-purpose flour', 'all purpose flour': 'maida / all-purpose flour', 
        'refined flour': 'maida / all-purpose flour',
        'gram flour': 'besan / gram flour', 'besan flour': 'besan / gram flour', 
        'whole wheat flour': 'wheat flour', 'atta': 'wheat flour',
        'rice flour': 'rice / rice flour', 'raw rice': 'rice / rice flour', 'idli rice': 'rice / rice flour', 
        'brown rice': 'rice / rice flour', 'sticky rice': 'rice / rice flour', 
        'forbidden black rice': 'rice / rice flour',
        'semolina': 'semolina / rava', 'rava': 'semolina / rava', 'sooji': 'semolina / rava',
        'urad dal': 'urad dal', 'split urad dal': 'urad dal', 'whole urad dal': 'urad dal', 
        'chana dal': 'dal / lentils', 'split pigeon peas': 'dal / lentils', 'arhar dal': 'dal / lentils', 
        'moong dal': 'dal / lentils', 'masoor dal': 'dal / lentils', 'toor dal': 'dal / lentils',
        'curd': 'yogurt / curd', 'yogurt': 'yogurt / curd', 'dahi': 'yogurt / curd',
        'milk powder': 'milk', 'condensed milk': 'milk', 'reduced milk': 'milk',
        'gur': 'jaggery', 
        'ginger paste': 'ginger', 'ginger powder': 'ginger', 
        'garlic paste': 'garlic', 'garlic powder': 'garlic',
        'cardamom pods': 'cardamom', 'green cardamom': 'cardamom',
        'rose water': 'rose extract',
        'chicken': 'meat', 'mutton': 'meat', 'pork': 'meat', 'lamb': 'meat', 'beef': 'meat',
        'fish': 'seafood', 'prawns': 'seafood', 'lobster': 'seafood', 'bombay duck': 'seafood',
        'cottage cheese': 'paneer / cheese', 'chenna': 'paneer / cheese', 'chhena': 'paneer / cheese', 
        'paneer': 'paneer / cheese',
        'jaggery': 'jaggery', 
        'kewra': 'kewra extract'
    }
    
    cleaned = []
    for ingredient in ingredient_list:
        ingredient = ingredient.lower().strip()
        ingredient = synonym_map.get(ingredient, ingredient)
        
        if ingredient not in ['and', 'a', 'of', '']:
            cleaned.append(ingredient)
    return cleaned

def load_data_and_train_model():
    """Loads the data and pre-calculates the Apriori rules once."""
    global df, transactions, df_encoded, RULES_DF

    try:
        # Load the dataset
        df = pd.read_csv('indian_food.csv', on_bad_lines='skip')
    except FileNotFoundError:
        print("ERROR: 'indian_food.csv' not found. Ensure it is in the project root directory.")
        return False
        
    df['ingredients'] = df['ingredients'].fillna('').apply(lambda x: clean_ingredients(x.split(',')))
    transactions = df['ingredients'].tolist()

    if not transactions or all(not t for t in transactions):
        print("WARNING: Transactions list is empty or invalid.")
        return False
        
    # APRIORI ALGORITHM EXECUTION
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(df_encoded, min_support=0.005, use_colnames=True)

    RULES_DF = association_rules(frequent_itemsets, metric="lift", min_threshold=1.05)
    RULES_DF = RULES_DF.sort_values(by=['confidence', 'lift'], ascending=False)
    print(f"INFO: {len(RULES_DF)} Association Rules generated.")
    
    return True

# Simple fallback co-occurrence count (from preprocess.py)
def fallback_suggest_pairings(input_ingredients, top_n=5):
    global transactions
    pairs = {}
    input_set = set(input_ingredients)
    
    for ingredients in transactions:
        if input_set.intersection(set(ingredients)):
            for pair in ingredients:
                if pair not in input_set:
                    pairs[pair] = pairs.get(pair, 0) + 1
                    
    sorted_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)
    return [(item[0], item[1]/len(transactions)) for item in sorted_pairs[:top_n]]

# Apriori-based Ingredient Pairing Suggestion (from preprocess.py)
def suggest_pairings_apriori(input_ingredients, top_n=10):
    global RULES_DF
    suggestions = {}
    input_set = set(input_ingredients)
    
    for index, rule in RULES_DF.iterrows():
        antecedent_set = set(rule['antecedents'])
        if antecedent_set.issubset(input_set):
            for consequent_ing in rule['consequents']:
                if consequent_ing not in input_set:
                    current_score = suggestions.get(consequent_ing, 0)
                    suggestions[consequent_ing] = max(current_score, rule['confidence'])
    
    sorted_suggestions = sorted(suggestions.items(), key=lambda item: item[1], reverse=True)
    
    if not sorted_suggestions:
        return fallback_suggest_pairings(input_ingredients, top_n)
        
    return sorted_suggestions[:top_n]


# Dish Recommendations (from preprocess.py)
def recommend_dishes(selected_dish, top_n=5):
    global df, df_encoded
    if df is None or df_encoded is None: return []

    selected_dish_clean = selected_dish.strip().lower()
    df['name_clean'] = df['name'].str.lower().str.strip()
    selected_ingredients = None
    source_name = selected_dish.title()
    
    # 1. Check if the input is a known INGREDIENT
    if selected_dish_clean in df_encoded.columns:
        ingredient_dishes = {}
        for index, row in df.iterrows():
            if selected_dish_clean in row['ingredients']:
                ingredient_dishes[row['name']] = len(row['ingredients'])
        sorted_dishes = sorted(ingredient_dishes.items(), key=lambda item: item[1], reverse=True)
        return {'source_name': selected_dish.title(), 'recommendations': [name for name, score in sorted_dishes[:top_n]]}

    # 2. Check for an EXACT DISH NAME match
    try:
        selected_ingredients = df.loc[df['name_clean'] == selected_dish_clean, 'ingredients'].iloc[0]
    except IndexError:
        # 3. FALLBACK: Check for PARTIAL DISH NAME match
        partial_matches = df[df['name_clean'].str.contains(selected_dish_clean, na=False)]
        
        if not partial_matches.empty:
            closest_dish_name = partial_matches['name'].iloc[0]
            selected_ingredients = partial_matches['ingredients'].iloc[0]
            selected_dish_clean = partial_matches['name_clean'].iloc[0]
            source_name = closest_dish_name.title()
        else:
            return {'source_name': selected_dish.title(), 'recommendations': []}
    
    # --- 4. Recommendation Logic (Uses ingredient overlap) ---
    dish_scores = {}
    
    for index, row in df.iterrows():
        dish_name = row['name']
        if row['name_clean'] == selected_dish_clean: continue
            
        current_dish_ingredients = row['ingredients']
        overlap = len(set(selected_ingredients).intersection(set(current_dish_ingredients)))
        
        if overlap > 0: dish_scores[dish_name] = overlap
            
    sorted_dishes = sorted(dish_scores.items(), key=lambda item: item[1], reverse=True)
    
    return {
        'source_name': source_name,
        'recommendations': [name for name, score in sorted_dishes[:top_n]]
    }


# Function for Meal Planner (Filter) (from preprocess.py)
def meal_planner(diet_type=None, region=None):
    global df
    if df is None: return ["Recommender data not loaded."]

    filtered_dishes = df.copy()
    
    diet_type_clean = diet_type.strip().lower() if diet_type else None
    region_clean = region.strip().lower() if region else None

    # Primary Filter
    if diet_type_clean:
        filtered_dishes = filtered_dishes[filtered_dishes['diet'].str.lower() == diet_type_clean]
    if region_clean:
        filtered_dishes = filtered_dishes[filtered_dishes['region'].str.lower() == region_clean]
    
    if filtered_dishes.empty:
        # Fallback to broader filter (e.g., just by region OR just by diet)
        filtered_dishes = df.copy()
        if region_clean:
             filtered_dishes = filtered_dishes[filtered_dishes['region'].str.lower() == region_clean]
        elif diet_type_clean:
            filtered_dishes = filtered_dishes[filtered_dishes['diet'].str.lower() == diet_type_clean]
        
        if filtered_dishes.empty:
            return ["No dishes found matching this general criteria."]

    return filtered_dishes['name'].tolist()

# Execute the setup function
load_data_and_train_model()