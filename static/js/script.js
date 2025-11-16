function navigateTo(page) {
    // Flask uses route names, so we need a more robust way to navigate
    // when clicking buttons in the non-Flask rendered index.html (as provided).
    // For the actual Flask routes, we rely on the Jinja2 tags in other HTML.
    
    // Simple mapping based on the provided button handlers in index.html
    const routeMap = {
        'ingredients.html': '/ingredients',
        'recipes.html': '/recipes',
        'mealplanner.html': '/mealplanner'
    };
    
    window.location.href = routeMap[page] || '/';
}