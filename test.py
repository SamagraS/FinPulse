#!/usr/bin/env python3
"""
Enhanced Model 2: Product Recommender with Multiple Input Sources
- 2 CSV files from Model 1 (customer segments + personas)
- 2 JSON files from Model 4 (bottleneck analysis + constraints)
- 1 CSV file (government schemes)
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Optional
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Configuration Constants ===
# Placeholder filenames (update as needed)
DATA_FOLDER = '.'  # assuming current folder; modify if needed
FILE_PATHS = {
    'customer_segments_file': os.path.join(DATA_FOLDER, 'customer_segmentation_output.csv'),
    'customer_personas_file': os.path.join(DATA_FOLDER, 'new_customer_segmentation_output.csv'),
    'bottleneck_analysis_file': os.path.join(DATA_FOLDER, 'model4_results_customer_segmentation_output.json'),
    'constraint_mapping_file': os.path.join(DATA_FOLDER, 'model4_results_new_customer_segmentation_output.json'),
    'govt_schemes_file': os.path.join(DATA_FOLDER, 'government_schemes.csv')
}

DEFAULTS = {
    'risk_appetite': 'moderate',
    'digital_literacy': 'medium',
    'trust_level': 'medium',
    'preferred_channels': 'branch',
    'priority_sample_customers': ['CUST_001', 'CUST_002', 'CUST_003'],
    'output_file': os.path.join(DATA_FOLDER, 'enhanced_model2_results.json')
}

CONSTRAINT_PENALTIES = {
    'AWARENESS': 0.8,
    'DOCUMENTATION': 0.6,
    'TRUST': 0.7,
    'ACCESSIBILITY': 0.5,
    'DIGITAL_READINESS': 0.4
}

RATIONALE_MAPPINGS = {
    'segment_fit': {
        ('agriculture', 'crop'): "Perfect match for agricultural needs and farming cycle",
        ('salaried', ('savings', 'insurance')): "Ideal for regular income earners with stable cash flow",
        ('senior', ('pension', 'healthcare')): "Specifically designed for senior citizen requirements",
    },
    'constraint_solutions': {
        'AWARENESS': "Targeted awareness campaigns can address {severity:.0%} knowledge gap",
        'DOCUMENTATION': "Simplified documentation process reduces {severity:.0%} complexity barrier",
        'TRUST': "Trust-building initiatives can overcome {severity:.0%} skepticism",
        'ACCESSIBILITY': "Mobile banking units improve {severity:.0%} access challenges",
        'DIGITAL_READINESS': "Assisted digital services bridge {severity:.0%} technology gap"
    },
    'financial': "Loan size manageable at {ratio:.1f}x monthly income",
    'subsidy': "Government subsidy reduces effective cost significantly",
    'insurance': "Essential risk protection with government backing"
}

class EnhancedProductRecommender:
    def __init__(self, 
                 customer_segments_file: str,
                 customer_personas_file: str,
                 bottleneck_analysis_file: str,
                 constraint_mapping_file: str,
                 govt_schemes_file: str):
        
        self.customer_segments_file = customer_segments_file
        self.customer_personas_file = customer_personas_file
        self.bottleneck_analysis_file = bottleneck_analysis_file
        self.constraint_mapping_file = constraint_mapping_file
        self.govt_schemes_file = govt_schemes_file
        
        self.load_all_data()
        
    def load_all_data(self):
        try:
            logger.info("Loading customer segments...")
            self.customer_segments = pd.read_csv(self.customer_segments_file)
            logger.info("Loading customer personas...")
            self.customer_personas = pd.read_csv(self.customer_personas_file)
            
            logger.info("Loading bottleneck analysis...")
            with open(self.bottleneck_analysis_file, 'r') as f:
                self.bottleneck_data = json.load(f)
                
            logger.info("Loading constraint mapping...")
            with open(self.constraint_mapping_file, 'r') as f:
                self.constraint_mapping = json.load(f)
                
            logger.info("Loading government schemes...")
            self.govt_schemes = pd.read_csv(self.govt_schemes_file)
            
            logger.info("All data loaded successfully.")
            self.validate_data_integrity()
        except Exception as e:
            logger.error(f"Error loading data files: {e}")
            raise
    
    def validate_data_integrity(self):
        required_segment_cols = ['customer_id', 'segment_name', 'monthly_income', 'age', 'occupation_category']
        missing = [col for col in required_segment_cols if col not in self.customer_segments.columns]
        if missing:
            logger.warning(f"Missing columns in customer segments: {missing}")
        
        required_scheme_cols = ['scheme_id', 'scheme_name', 'category', 'min_amount', 'max_amount', 'eligibility_criteria']
        missing = [col for col in required_scheme_cols if col not in self.govt_schemes.columns]
        if missing:
            logger.warning(f"Missing columns in government schemes: {missing}")
        
        logger.info(f"Data Summary: Customer Segments({len(self.customer_segments)}), Personas({len(self.customer_personas)}), Government Schemes({len(self.govt_schemes)}), Bottlenecks({len(self.bottleneck_data)})")
    
    def get_customer_profile(self, customer_id: str) -> Optional[Dict]:
        segment_data = self.customer_segments[self.customer_segments['customer_id'] == customer_id]
        if segment_data.empty:
            logger.warning(f"Customer {customer_id} not found in segments")
            return None
        customer = segment_data.iloc[0].to_dict()
        
        segment_name = customer.get('segment_name', '')
        persona_data = self.customer_personas[self.customer_personas['segment_name'] == segment_name]
        if not persona_data.empty:
            persona_info = persona_data.iloc[0].to_dict()
            customer.update({
                'persona_risk_appetite': persona_info.get('risk_appetite', DEFAULTS['risk_appetite']),
                'persona_digital_literacy': persona_info.get('digital_literacy', DEFAULTS['digital_literacy']),
                'persona_trust_level': persona_info.get('trust_level', DEFAULTS['trust_level']),
                'persona_preferred_channels': persona_info.get('preferred_channels', DEFAULTS['preferred_channels'])
            })
        return customer
    
    def get_constraint_analysis(self, segment_name: str, customer_profile: Dict) -> Dict:
        segment_key = segment_name.lower().replace(' ', '').replace('-', '')
        default_constraint = {
            "primary_constraint": {"type": "AWARENESS", "severity": 0.6, "confidence": 0.7, "impact_potential": 1.2},
            "secondary_constraints": [{"type": "DOCUMENTATION", "severity": 0.4, "likelihood_of_emergence": 0.5}],
            "throughput_analysis": {"current_throughput": 0.3, "improvement_mean": 0.4, "new_throughput_expected": 0.7, "probability_of_30pct_improvement": 0.8}
        }
        for analysis_key, analysis_data in self.bottleneck_data.items():
            if segment_key in analysis_key.lower() or any(word in analysis_key.lower() for word in segment_name.lower().split()):
                return analysis_data
        
        constraint_adjustments = self.constraint_mapping.get('segment_adjustments', {})
        if customer_profile.get('age', 35) > 50:
            default_constraint['primary_constraint']['type'] = 'DIGITAL_READINESS'
            default_constraint['primary_constraint']['severity'] += 0.2
        if customer_profile.get('monthly_income', 25000) < 20000:
            if 'TRUST' not in [c['type'] for c in default_constraint['secondary_constraints']]:
                default_constraint['secondary_constraints'].append({"type": "TRUST", "severity": 0.5, "likelihood_of_emergence": 0.6})
        return default_constraint
    
    def check_scheme_eligibility(self, customer: Dict, scheme_row: pd.Series) -> bool:
        eligibility_criteria = str(scheme_row.get('eligibility_criteria', '')).lower()
        
        if 'age_min' in scheme_row and not pd.isna(scheme_row['age_min']):
            if customer.get('age', 0) < scheme_row['age_min']:
                return False
        if 'age_max' in scheme_row and not pd.isna(scheme_row['age_max']):
            if customer.get('age', 100) > scheme_row['age_max']:
                return False
        if 'income_max' in scheme_row and not pd.isna(scheme_row['income_max']):
            if customer.get('monthly_income', 0) > scheme_row['income_max']:
                return False
        
        customer_occupation = str(customer.get('occupation_category', '')).lower()
        if 'agriculture' in eligibility_criteria and 'agriculture' not in customer_occupation:
            if customer_occupation not in ['farmer', 'agriculture']:
                return False
        if 'salaried' in eligibility_criteria and 'salaried' not in customer_occupation:
            if customer_occupation not in ['salaried', 'employed']:
                return False
        
        if 'rural' in eligibility_criteria:
            segment = str(customer.get('segment_name', '')).lower()
            if 'rural' not in segment and 'remote' not in segment:
                return False
        
        return True
    
    def calculate_propensity_score(self, customer: Dict, scheme: pd.Series, constraint_analysis: Dict) -> float:
        base_score = 0.5
        scheme_category = str(scheme.get('category', '')).lower()
        customer_segment = str(customer.get('segment_name', '')).lower()
        
        if 'agriculture' in customer_segment or 'farmer' in customer_segment:
            if scheme_category in ['loan', 'insurance', 'subsidy']:
                base_score += 0.3
            if 'crop' in str(scheme.get('scheme_name', '')).lower():
                base_score += 0.2
        if 'salaried' in customer_segment:
            if scheme_category in ['savings', 'insurance']:
                base_score += 0.2
        if 'senior' in customer_segment:
            if scheme_category in ['pension', 'healthcare']:
                base_score += 0.3
        
        income = customer.get('monthly_income', 25000)
        scheme_min = scheme.get('min_amount', 0)
        scheme_max = scheme.get('max_amount', 100000)
        
        if scheme_category == 'loan':
            if income * 6 >= scheme_min:
                base_score += 0.1
            if income < 30000:
                base_score += 0.15
        
        primary_constraint = constraint_analysis.get('primary_constraint', {})
        constraint_type = primary_constraint.get('type', 'AWARENESS')
        severity = primary_constraint.get('severity', 0.5)
    
        penalty_factor = CONSTRAINT_PENALTIES.get(constraint_type, 0.6)
        constraint_penalty = severity * penalty_factor
        final_score = base_score * (1 - constraint_penalty)
    
        return max(0.1, min(1.0, final_score))
    
    def calculate_ticket_size(self, customer: Dict, scheme: pd.Series) -> float:
        income = customer.get('monthly_income', 25000)
        scheme_category = str(scheme.get('category', '')).lower()
        min_amount = scheme.get('min_amount', 0)
        max_amount = scheme.get('max_amount', 100000)
        
        if scheme_category == 'loan':
            recommended = min(income * 6, max_amount)
            return max(recommended, min_amount)
        elif scheme_category == 'insurance':
            recommended = min(income * 0.8, max_amount) 
            return max(recommended, min_amount)
        elif scheme_category == 'savings':
            recommended = min(income * 0.15, max_amount)
            return max(recommended, min_amount)
        else:
            recommended = min(income * 2, max_amount)
            return max(recommended, min_amount)
    
    def build_recommendation_rationale(self, customer: Dict, scheme: pd.Series, constraint_analysis: Dict, propensity: float) -> List[str]:
        rationale = []
        segment = str(customer.get('segment_name', '')).lower()
        scheme_name = scheme.get('scheme_name', '')
        scheme_category = str(scheme.get('category', '')).lower()
        
        for (seg_key, cat_key), message in RATIONALE_MAPPINGS['segment_fit'].items():
            if (seg_key in segment) and (cat_key == '' or (scheme_category if isinstance(cat_key, str) else any(c in scheme_category for c in cat_key))):
                rationale.append(message)
        
        primary_constraint = constraint_analysis.get('primary_constraint', {})
        constraint_type = primary_constraint.get('type', 'AWARENESS')
        severity = primary_constraint.get('severity', 0.5)
        
        if constraint_type in RATIONALE_MAPPINGS['constraint_solutions']:
            rationale.append(RATIONALE_MAPPINGS['constraint_solutions'][constraint_type].format(severity=severity))
        
        income = customer.get('monthly_income', 25000)
        ticket_size = self.calculate_ticket_size(customer, scheme)
        
        if scheme_category == 'loan' and ticket_size <= income * 8:
            ratio = ticket_size / income
            rationale.append(RATIONALE_MAPPINGS['financial'].format(ratio=ratio))
        
        if 'subsidy' in scheme_name.lower():
            rationale.append(RATIONALE_MAPPINGS['subsidy'])
        
        if 'insurance' in scheme_category:
            rationale.append(RATIONALE_MAPPINGS['insurance'])
        
        return rationale[:4]
    
    def generate_recommendations(self, customer_id: str, top_n: int = 5) -> Dict:
        customer = self.get_customer_profile(customer_id)
        if not customer:
            return {"error": f"Customer {customer_id} not found"}
        
        segment_name = customer.get('segment_name', '')
        constraint_analysis = self.get_constraint_analysis(segment_name, customer)
        
        logger.info(f"Processing {customer_id} - {segment_name}, Primary Constraint: {constraint_analysis['primary_constraint']['type']} "
                    f"({constraint_analysis['primary_constraint']['severity']:.1%} severity)")
        
        recommendations = []
        for _, scheme_row in self.govt_schemes.iterrows():
            if not self.check_scheme_eligibility(customer, scheme_row):
                continue
            
            propensity = self.calculate_propensity_score(customer, scheme_row, constraint_analysis)
            ticket_size = self.calculate_ticket_size(customer, scheme_row)
            profitability = scheme_row.get('profitability_score', 0.6)
            throughput = propensity * ticket_size * profitability
            
            rationale = self.build_recommendation_rationale(customer, scheme_row, constraint_analysis, propensity)
            
            recommendations.append({
                'scheme_id': scheme_row.get('scheme_id'),
                'scheme_name': scheme_row.get('scheme_name'),
                'category': scheme_row.get('category'),
                'recommended_amount': int(ticket_size),
                'propensity_score': round(propensity, 3),
                'throughput_contribution': int(throughput),
                'constraint_impact': constraint_analysis['primary_constraint']['type'],
                'constraint_severity': round(constraint_analysis['primary_constraint']['severity'], 3),
                'rationale': rationale,
                'priority_score': round(propensity * 0.7 + (throughput / 100000) * 0.3, 3)
            })
        
        recommendations.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return {
            'customer_id': customer_id,
            'customer_segment': segment_name,
            'customer_profile': {
                'monthly_income': customer.get('monthly_income'),
                'age': customer.get('age'),
                'occupation': customer.get('occupation_category'),
                'risk_appetite': customer.get('persona_risk_appetite', DEFAULTS['risk_appetite']),
                'digital_literacy': customer.get('persona_digital_literacy', DEFAULTS['digital_literacy'])
            },
            'constraint_analysis': {
                'primary_constraint': constraint_analysis['primary_constraint']['type'],
                'constraint_severity': constraint_analysis['primary_constraint']['severity'],
                'confidence': constraint_analysis['primary_constraint']['confidence'],
                'throughput_improvement_potential': constraint_analysis['throughput_analysis']['improvement_mean']
            },
            'recommendations': recommendations[:top_n],
            'total_eligible_schemes': len(recommendations),
            'total_potential_throughput': sum(r['throughput_contribution'] for r in recommendations),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def batch_process_customers(self, customer_ids: Optional[List[str]] = None, top_n: int = 5) -> List[Dict]:
        if customer_ids is None:
            customer_ids = self.customer_segments['customer_id'].tolist()
        logger.info(f"Processing {len(customer_ids)} customers in batch...")
        
        results = []
        for cid in customer_ids:
            try:
                results.append(self.generate_recommendations(cid, top_n))
            except Exception as e:
                logger.error(f"Error processing customer {cid}: {e}")
                results.append({"customer_id": cid, "error": str(e)})
        return results

def main():
    print("=" * 80)
    print("🏦 ENHANCED MODEL 2: MULTI-INPUT PRODUCT RECOMMENDER")
    print("=" * 80)
    
    try:
        recommender = EnhancedProductRecommender(**FILE_PATHS)
        results = recommender.batch_process_customers(DEFAULTS['priority_sample_customers'], top_n=3)

        for result in results:
            if 'error' in result:
                print(f"\n❌ Error for {result['customer_id']}: {result['error']}")
                continue

            print(f"\n" + "=" * 60)
            print(f"📋 RECOMMENDATIONS FOR {result['customer_id']}")
            print(f"Segment: {result['customer_segment']}")
            print(f"Income: ₹{result['customer_profile']['monthly_income']:,}/month")
            print(f"Primary Constraint: {result['constraint_analysis']['primary_constraint']} ({result['constraint_analysis']['constraint_severity']:.1%} severity)")

            print(f"\n🏆 TOP {len(result['recommendations'])} SCHEME RECOMMENDATIONS:")
            for i, rec in enumerate(result['recommendations'], 1):
                print(f"\n   {i}. {rec['scheme_name']}")
                print(f"      🏷  Category: {rec['category'].upper()}")
                print(f"      💰 Recommended Amount: ₹{rec['recommended_amount']:,}")
                print(f"      📈 Propensity Score: {rec['propensity_score']:.1%}")
                print(f"      🎯 Throughput: ₹{rec['throughput_contribution']:,}")
                print(f"      ⚠  Constraint: {rec['constraint_impact']} ({rec['constraint_severity']:.1%})")
                print(f"      💡 Rationale:")
                for rationale in rec['rationale']:
                    print(f"        • {rationale}")

            print(f"\n📊 Summary:")
            print(f"   • {result['total_eligible_schemes']} eligible schemes")
            print(f"   • ₹{result['total_potential_throughput']:,} total throughput potential")

        output_file = DEFAULTS['output_file']
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_file}")
        return results
    
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()