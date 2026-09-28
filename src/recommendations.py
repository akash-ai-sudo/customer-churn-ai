def get_recommendation(customer_data, risk_level):
    """
    Generates a simple retention recommendation based on customer attributes and risk.
    """
    if risk_level == "Low Risk":
        return "Customer is loyal. Regular engagement and routine check-ins recommended."
        
    recommendations = []
    
    # Basic rule-based recommendations on significant cost/service features
    if customer_data.get('Contract') == 'Month-to-month':
        recommendations.append("Offer a discount to upgrade to a 1-year or 2-year contract.")
        
    if customer_data.get('InternetService') == 'Fiber optic':
        recommendations.append("Review Fiber optic service quality or offer a bundled tech-support package.")
        
    if float(customer_data.get('MonthlyCharges', 0)) > 70:
        recommendations.append("High monthly charges detected. Consider offering a customized loyalty discount or reasonable alternative package.")
        
    if customer_data.get('TechSupport') == 'No' and customer_data.get('InternetService') != 'No':
        recommendations.append("Provide a free trial of Tech Support. Lack of support often frustrates users.")
        
    if not recommendations:
        recommendations.append("Reach out for a customer feedback call to understand their pain points.")
        
    return " ".join(recommendations)
