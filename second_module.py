import pandas as pd
import numpy as np
from typing import Dict, List

class SimpleProductRecommender:
    def __init__(self):
        # self.products = pd.read_csv(products_file)
        # self.eligibility = pd.read_csv(eligibility_file)
        # self.constraints = pd.read_csv(constraints_file)
        self.products = pd.DataFrame([
            {'product_id': 'P001', 'product_name': 'Crop Loan - Rabi', 'category': 'loan', 'min_amount': 10000, 'max_amount': 50000, 'profitability_score': 1.2},
            {'product_id': 'P002', 'product_name': 'Crop Insurance', 'category': 'insurance', 'min_amount': 100000, 'max_amount': 200000, 'profitability_score': 1.0},
            {'product_id': 'P003', 'product_name': 'Mudra Loan - Kishor', 'category': 'loan', 'min_amount': 50000, 'max_amount': 100000, 'profitability_score': 1.1},
            {'product_id': 'P004', 'product_name': 'Postal Life Insurance', 'category': 'insurance', 'min_amount': 100000, 'max_amount': 500000, 'profitability_score': 0.9},
            {'product_id': 'P005', 'product_name': 'Public Provident Fund', 'category': 'savings', 'min_amount': 50000, 'max_amount': 150000, 'profitability_score': 1.3},
            {'product_id': 'P006', 'product_name': 'National Savings Certificate', 'category': 'savings', 'min_amount': 15000, 'max_amount': 50000, 'profitability_score': 1.1},
            {'product_id': 'P007', 'product_name': 'Mudra Loan - Tarun', 'category': 'loan', 'min_amount': 500000, 'max_amount': 800000, 'profitability_score': 1.2},
            {'product_id': 'P008', 'product_name': 'Business Insurance', 'category': 'insurance', 'min_amount': 500000, 'max_amount': 1000000, 'profitability_score': 1.0},
            {'product_id': 'P009', 'product_name': 'Mudra Loan - Shishu', 'category': 'loan', 'min_amount': 20000, 'max_amount': 30000, 'profitability_score': 1.0},
            {'product_id': 'P010', 'product_name': 'Artisan Insurance Scheme', 'category': 'insurance', 'min_amount': 50000, 'max_amount': 100000, 'profitability_score': 1.0},
            {'product_id': 'P011', 'product_name': 'Post Office Savings Account', 'category': 'savings', 'min_amount': 10000, 'max_amount': 30000, 'profitability_score': 1.2},
        ])

        # Hardcoded eligibility rules (simplified)
        self.eligibility = pd.DataFrame([
            {'product_id': 'P001', 'field_name': 'current_month', 'condition_type': 'list', 'required_values': '10|11|12'},
            {'product_id': 'P002', 'field_name': 'existing_loans', 'condition_type': 'min', 'min_value': 0},
            {'product_id': 'P003', 'field_name': 'segment_name', 'condition_type': 'list', 'required_values': 'Rural Women Entrepreneurs'},
            # Other rules can be added as needed
        ])

        # Hardcoded constraints (simplified)
        self.constraints = pd.DataFrame([
            {'product_id': 'P001', 'constraint_type': 'awareness', 'impact_factor': 0.3},
            {'product_id': 'P002', 'constraint_type': 'documentation', 'impact_factor': 0.2},
            {'product_id': 'P003', 'constraint_type': 'documentation', 'impact_factor': 0.25},
            {'product_id': 'P004', 'constraint_type': 'awareness', 'impact_factor': 0.15},
            {'product_id': 'P005', 'constraint_type': 'digital_literacy_gap', 'impact_factor': 0.1},
            {'product_id': 'P006', 'constraint_type': 'trust_deficit', 'impact_factor': 0.2},
            {'product_id': 'P007', 'constraint_type': 'seasonal_cashflow', 'impact_factor': 0.3},
            {'product_id': 'P008', 'constraint_type': 'seasonal_cashflow', 'impact_factor': 0.25},
            {'product_id': 'P009', 'constraint_type': 'awareness', 'impact_factor': 0.35},
            {'product_id': 'P010', 'constraint_type': 'awareness', 'impact_factor': 0.2},
            {'product_id': 'P011', 'constraint_type': 'trust_deficit', 'impact_factor': 0.1},
        ])
        
    def check_eligibility(self, customer: dict, product_id: str) -> bool:
        """Check if customer is eligible for product"""
        product_rules = self.eligibility[self.eligibility['product_id'] == product_id]
        
        for _, rule in product_rules.iterrows():
            field = rule['field_name']
            if field not in customer:
                continue
                
            customer_value = customer[field]
            
            if rule['condition_type'] == 'range':
                if customer_value < rule['min_value'] or customer_value > rule['max_value']:
                    return False
            elif rule['condition_type'] == 'min':
                if customer_value < rule['min_value']:
                    return False
            elif rule['condition_type'] == 'list':
                required_vals = str(rule['required_values']).split('|')
                if customer_value not in required_vals:
                    return False
        
        return True
    
    def calculate_propensity(self, customer: dict, product_id: str) -> float:
        """Simple rule-based propensity calculation"""
        base_score = 0.5
        
        product = self.products[self.products['product_id'] == product_id].iloc[0]
        
        # Income-based scoring
        income = customer.get('monthly_income', 25000)
        if product['category'] == 'loan':
            if income > 30000:
                base_score += 0.2
            elif income < 15000:
                base_score -= 0.1
        
        # Segment-based scoring (from Model 1)
        segment_name = customer.get('segment_name', '')
        if 'farmer' in segment_name.lower() and product['category'] == 'loan':
            base_score += 0.3
        if 'senior' in segment_name.lower() and product['category'] == 'savings':
            base_score += 0.2
        if 'entrepreneur' in segment_name.lower() and 'mudra' in product['product_name'].lower():
            base_score += 0.3
        
        # Risk appetite matching
        risk_appetite = customer.get('risk_appetite', 'moderate')
        if product['category'] == 'investment':
            if risk_appetite == 'high':
                base_score += 0.1
            elif risk_appetite == 'low':
                base_score -= 0.2
        
        return min(1.0, max(0.0, base_score))
    
    def apply_constraint_penalty(self, propensity: float, product_id: str, 
                                primary_constraint: str, severity: float) -> float:
        """Apply constraint-based penalty"""
        product_constraints = self.constraints[self.constraints['product_id'] == product_id]
        
        for _, constraint in product_constraints.iterrows():
            if constraint['constraint_type'] == primary_constraint:
                penalty = severity * constraint['impact_factor']
                propensity = propensity * (1 - penalty)
                break
        
        return propensity
    
    def calculate_ticket_size(self, customer: dict, product_id: str) -> float:
        """Calculate appropriate amount for customer"""
        product = self.products[self.products['product_id'] == product_id].iloc[0]
        income = customer.get('monthly_income', 25000)
        
        if product['category'] == 'loan':
            # Loan: 3-6x monthly income, capped by product limits
            amount = min(income * 4, product['max_amount'])
            return max(amount, product['min_amount'])
        elif product['category'] == 'insurance':
            # Insurance: 10-20x monthly income
            amount = min(income * 15, product['max_amount'])
            return max(amount, product['min_amount'])
        else:
            # Savings/Investment: Based on disposable income
            amount = min(income * 0.3, product['max_amount'])
            return max(amount, product['min_amount'])
    
    def recommend(self, customer: dict, model4_output: dict, top_n: int = 3) -> dict:
        """Main recommendation function"""
        recommendations = []
        
        # Extract constraint info from Model 4
        primary_constraint = model4_output.get('primary_constraint', 'awareness')
        severity = model4_output.get('severity', 0.5)
        
        for _, product in self.products.iterrows():
            product_id = product['product_id']
            
            # Check eligibility
            if not self.check_eligibility(customer, product_id):
                continue
            
            # Calculate base propensity
            propensity = self.calculate_propensity(customer, product_id)
            
            # Apply constraint penalty
            adjusted_propensity = self.apply_constraint_penalty(
                propensity, product_id, primary_constraint, severity
            )
            
            # Calculate ticket size
            amount = self.calculate_ticket_size(customer, product_id)
            
            # Calculate throughput contribution
            throughput = adjusted_propensity * amount * product['profitability_score']
            
            recommendations.append({
                'product_id': product_id,
                'product_name': product['product_name'],
                'category': product['category'],
                'recommended_amount': int(amount),
                'propensity_score': round(adjusted_propensity, 3),
                'throughput_contribution': int(throughput),
                'rationale': self.build_rationale(customer, product, primary_constraint)
            })
        
        # Sort by throughput contribution and return top N
        recommendations.sort(key=lambda x: x['throughput_contribution'], reverse=True)
        
        return {
            'customer_id': customer['customer_id'],
            'primary_constraint': primary_constraint,
            'recommendations': recommendations[:top_n],
            'total_eligible_products': len(recommendations)
        }
    
    def build_rationale(self, customer: dict, product: pd.Series, constraint: str) -> List[str]:
        """Build explanation for recommendation"""
        rationale = []
        
        # Segment matching
        segment_name = customer.get('segment_name', '')
        if 'farmer' in segment_name.lower() and 'crop' in product['product_name'].lower():
            rationale.append("Matches agricultural occupation")
        
        # Constraint addressing
        if constraint == 'awareness':
            rationale.append("Addresses awareness gap through targeted outreach")
        elif constraint == 'documentation':
            rationale.append("Simplified documentation process available")
        
        # Seasonal relevance (simple rule)
        if 'kharif' in product['product_name'].lower():
            rationale.append("Suitable for Kharif season (Apr-Jul)")
        elif 'rabi' in product['product_name'].lower():
            rationale.append("Suitable for Rabi season (Oct-Jan)")
        
        return rationale[:3]  # Limit to 3 reasons

# Usage Example
# def run_model2_example():
#     # Initialize recommender
#     recommender = SimpleProductRecommender(
#         'products.csv', 
#         'eligibility.csv', 
#         'constraints.csv'
#     )
    
#     # Sample customer (from Model 1)
#     customer = {
#         'customer_id': 'CUST_001',
#         'age': 35,
#         'monthly_income': 25000,
#         'occupation_category': 'agriculture',
#         'land_ownership': 2.5,
#         'segment_name': 'Middle-aged Agriculture - Rural Remote',
#         'risk_appetite': 'moderate'
#     }
    
#     # Sample Model 4 output
#     model4_output = {
#         'primary_constraint': 'awareness',
#         'severity': 0.7,
#         'impact_potential': 0.8
#     }
    
#     # Get recommendations
#     result = recommender.recommend(customer, model4_output, top_n=3)
    
#     return result
def scenario_1_rural_farmer():
    customer = {
        'customer_id': 'CUST_001',
        'age': 35,
        'monthly_income': 25000,
        'occupation_category': 'agriculture',
        'land_ownership': 2.5,
        'crop_types': 'cereal',
        'seasonal_income_variance': 0.7,
        'risk_appetite': 'moderate',
        'digital_literacy': 0.2,
        'existing_loans': 0,
        'existing_insurance': 1,
        'segment_name': 'Subsistence Rural Farmers',
        'trust_requirement': 'high',
        'financial_literacy_score': 0.3,
        'current_month': 10  # October - Rabi sowing season
    }
    model4_output = {
        'primary_constraint': 'awareness',
        'severity': 0.78,
        'impact_potential': 0.85,
        'expected_improvement': 0.45
    }
    return customer, model4_output

# Run example
if __name__ == "__main__":
    # result = run_model2_example()
    # print("Recommendations:", result)
    recommender = SimpleProductRecommender()
    customer, model4_output = scenario_1_rural_farmer()
    result = recommender.recommend(customer, model4_output, top_n=3)
    print("Recommendations for Scenario 1:")
    for rec in result['recommendations']:
        print(f"- {rec['product_name']}: Amount={rec['recommended_amount']}, Propensity={rec['propensity_score']}, Rationale={rec['rationale']}")