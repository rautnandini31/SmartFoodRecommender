# app.py

from flask import Flask, render_template, request, jsonify, redirect, url_for
from recommender import clean_ingredients, suggest_pairings_apriori, recommend_dishes, meal_planner
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, 'static'),
    template_folder=os.path.join(BASE_DIR, 'templates')
)

# --- View Routes (render HTML pages) ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ingredients')
def ingredients():
    return render_template('ingredients.html')

@app.route('/recipes')
def recipes():
    return render_template('recipes.html')

@app.route('/mealplanner')
def mealplanner():
    return render_template('mealplanner.html')

# --- API Endpoints (handle AJAX requests) ---

@app.route('/api/suggest_pairings', methods=['POST'])
def api_suggest_pairings():
    data = request.get_json()
    raw_ingredients = data.get('ingredients', '')
    
    ingredients_list = raw_ingredients.split(',')
    cleaned_ingredients = clean_ingredients(ingredients_list)
    
    if not cleaned_ingredients:
        return jsonify({'success': False, 'message': 'Please enter valid ingredients.'}), 400

    suggestions = suggest_pairings_apriori(cleaned_ingredients)
    
    formatted_suggestions = [
        {'ingredient': ing.title(), 'score': f'{score:.2f}'}
        for ing, score in suggestions
    ]
    
    return jsonify({
        'success': True,
        'results': formatted_suggestions,
        'message': f"Top {len(formatted_suggestions)} ingredient pairings suggested."
    })

@app.route('/api/recommend_dishes', methods=['POST'])
def api_recommend_dishes():
    data = request.get_json()
    selected_dish_or_ingredient = data.get('input', '').strip()
    
    if not selected_dish_or_ingredient:
        return jsonify({'success': False, 'message': 'Please enter a dish name or key ingredient.'}), 400
        
    result = recommend_dishes(selected_dish_or_ingredient)

    recommendations = result.get('recommendations', [])
    source_name = result.get('source_name', selected_dish_or_ingredient.title())

    if not recommendations:
        message = f"Could not find any dish related to '{source_name}'. Please check your input."
    elif selected_dish_or_ingredient.lower().strip() == source_name.lower().strip():
        # Input was an ingredient and matches the source name
        message = f"Top dishes containing '{source_name}':"
    else:
        # Input was a dish (or partially matched)
        message = f"Dishes similar to {source_name}:"
         
    return jsonify({
        'success': True,
        'message': message,
        'results': recommendations
    })

@app.route('/api/meal_planner', methods=['POST'])
def api_meal_planner():
    data = request.get_json()
    diet_type = data.get('diet', '').strip()
    region = data.get('region', '').strip()
    
    if not diet_type and not region:
        return jsonify({'success': False, 'message': 'Please select a Diet Type and/or a Region.'}), 400
        
    suggestions = meal_planner(diet_type, region)

    if not suggestions or suggestions[0] == "No dishes found matching this general criteria.":
        message = "No dishes found matching your criteria. Try broadening your search."
        suggestions = []
    else:
        message = f"Found {len(suggestions)} dishes matching your criteria:"
        
    return jsonify({
        'success': True,
        'message': message,
        'results': suggestions
    })


if __name__ == '__main__':
    import os
    
    # Use absolute path to the CSV so the check is accurate
    csv_path = os.path.join(BASE_DIR, 'indian_food.csv')
    
    if not os.path.exists(csv_path):
        print("\n" + "="*70)
        print("CRITICAL ERROR: 'indian_food.csv' not found at:", csv_path)
        print("Please place the CSV file in the project root directory (same folder as app.py).")
        print("="*70 + "\n")
    else:
        # Helpful debug prints
        print("Starting Flask app.")
        print("Project folder:", BASE_DIR)
        print("Static folder:", app.static_folder)
        print("Template folder:", app.template_folder)

        # Use 0.0.0.0 for deployment; default to 127.0.0.1 during local dev
        # Use dynamic port assigned by Render (or default to 5000 locally)
        port = int(os.environ.get("PORT", 5000))
        app.run(debug=True, host='0.0.0.0', port=port)
