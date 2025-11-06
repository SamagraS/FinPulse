import pandas as pd
import numpy as np
import json
import random
import os
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration: Can be moved to external JSON/YAML for tuning
CONSTRAINT_CATEGORIES = {
    'AWARENESS': {
        'indicators': ['product_awareness_score', 'past_campaign_response_rate'],
        'behavioral_signals': ['govt_scheme_preference', 'product_openness_score'],
        'customer_signals': ['channel_preference_digital_ratio', 'avg_segment_confidence'],
        'threshold': 0.4,
        'weight': 1.4,
        'impact_multiplier': 1.6
    },
    'ACCESSIBILITY': {
        'indicators': ['bank_branches_per_1000', 'agent_availability_score', 'road_connectivity_index'],
        'behavioral_signals': ['digital_acceptance'],
        'customer_signals': ['rural_customer_ratio', 'field_visit_preference_ratio'],
        'threshold': 0.5,
        'weight': 1.1,
        'impact_multiplier': 1.2
    },
    'DOCUMENTATION': {
        'indicators': ['kyc_completion_rate', 'avg_processing_days'],
        'behavioral_signals': ['documentation_comfort'],
        'customer_signals': ['high_doc_support_ratio', 'low_digital_literacy_ratio'],
        'threshold': 0.6,
        'weight': 1.0,
        'impact_multiplier': 1.0
    },
    'TRUST': {
        'indicators': ['customer_complaint_rate', 'application_approval_rate'],
        'behavioral_signals': ['trust_requirement', 'community_influence_level'],
        'customer_signals': ['high_trust_requirement_ratio', 'relationship_building_essential_ratio'],
        'threshold': 0.7,
        'weight': 1.2,
        'impact_multiplier': 1.3
    },
    'DIGITAL_READINESS': {
        'indicators': ['mobile_network_coverage', 'electricity_availability'],
        'behavioral_signals': ['digital_acceptance'],
        'customer_signals': ['low_digital_literacy_ratio', 'digital_channel_preference_ratio'],
        'threshold': 0.5,
        'weight': 0.8,
        'impact_multiplier': 0.9
    }
}

def generate_synthetic_regional_data(region_id: str = "REGION_XYZ") -> Dict[str, Any]:
    """Generate random but realistic regional dataset"""
    return {
        'region_id': region_id,
        'behavioral_aggregates': {
            'avg_trust_requirement': round(random.uniform(0.2, 0.9), 2),
            'avg_documentation_comfort': round(random.uniform(0.1, 0.7), 2),
            'avg_digital_acceptance': round(random.uniform(0.1, 0.8), 2),
            'avg_govt_scheme_preference': round(random.uniform(0.3, 0.95), 2),
            'avg_community_influence_level': round(random.uniform(0.2, 0.9), 2),
            'avg_price_sensitivity': round(random.uniform(0.3, 0.85), 2)
        },
        'infrastructure_metrics': {
            'bank_branches_per_1000': round(random.uniform(0.2, 0.9), 2),
            'post_office_density': round(random.uniform(0.5, 3.0), 2),
            'mobile_network_coverage': round(random.uniform(0.3, 0.95), 2),
            'road_connectivity_index': round(random.uniform(0.2, 0.9), 2),
            'electricity_availability': round(random.uniform(0.4, 1.0), 2)
        },
        'service_metrics': {
            'kyc_completion_rate': round(random.uniform(0.2, 0.9), 2),
            'application_approval_rate': round(random.uniform(0.3, 0.95), 2),
            'avg_processing_days': random.randint(5, 25),
            'customer_complaint_rate': round(random.uniform(0.01, 0.15), 2),
            'agent_availability_score': round(random.uniform(0.2, 0.9), 2)
        },
        'market_context': {
            'product_awareness_score': round(random.uniform(0.1, 0.8), 2),
            'competitor_presence': round(random.uniform(0.1, 0.7), 2),
            'past_campaign_response_rate': round(random.uniform(0.05, 0.5), 2),
            'seasonal_demand_volatility': round(random.uniform(0.2, 0.9), 2),
            'local_economic_stress_index': round(random.uniform(0.2, 0.9), 2)
        }
    }

def load_customer_data_from_csv(csv_file_path: str) -> List[Dict[str, Any]]:
    """Load and parse customer segmentation data from a CSV file"""
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        logger.error(f"Error loading customer CSV {csv_file_path}: {e}")
        return []
    
    customers = []
    for _, row in df.iterrows():
        customers.append(_parse_customer_row(row))
    
    logger.info(f"Successfully loaded {len(customers)} customer records from {csv_file_path}")
    return customers

def _parse_customer_row(row: pd.Series) -> Dict[str, Any]:
    """Helper to parse a CSV row into structured customer data"""
    def get_float(key: str, default: float) -> float:
        try:
            return float(row.get(key, default))
        except Exception:
            return default
    
    customer = {
        'customer_id': row.get('customer_id', 'UNKNOWN'),
        'cluster_id': row.get('cluster_id', 0),
        'segment_name': row.get('segment_name', 'Unknown Segment'),
        'segment_confidence': get_float('segment_confidence', 0.5),
        'segment_profile': {
            'avg_age': get_float('segment_profile.avg_age', 35),
            'avg_income': get_float('segment_profile.avg_income', 30000),
            'primary_location_type': row.get('segment_profile.primary_location_type', 'urban'),
            'primary_occupation': row.get('segment_profile.primary_occupation', 'salaried'),
            'digital_literacy_level': get_float('segment_profile.digital_literacy_level', 0.5),
            'risk_appetite': row.get('segment_profile.risk_appetite', 'moderate'),
            'income_stability': get_float('segment_profile.income_stability', 0.7),
            'financial_literacy': get_float('segment_profile.financial_literacy', 0.5),
        },
        'individual_traits': {
            'age_vs_segment': row.get('individual_traits.age_vs_segment', 'typical'),
            'income_vs_segment': row.get('individual_traits.income_vs_segment', 'typical'),
            'digital_vs_segment': row.get('individual_traits.digital_vs_segment', 'typical'),
            'risk_vs_segment': row.get('individual_traits.risk_vs_segment', 'typical'),
        },
        'predicted_behaviors': {
            'product_openness': row.get('predicted_behaviors.product_openness', 'moderate'),
            'channel_preference': row.get('predicted_behaviors.channel_preference', 'in_person'),
            'decision_speed': row.get('predicted_behaviors.decision_speed', 'moderate'),
            'price_sensitivity': row.get('predicted_behaviors.price_sensitivity', 'moderate'),
            'trust_requirement': row.get('predicted_behaviors.trust_requirement', 'moderate'),
        },
        'engagement_strategy': {
            'primary_channel': row.get('engagement_strategy.primary_channel', 'phone'),
            'message_complexity': row.get('engagement_strategy.message_complexity', 'moderate'),
            'relationship_building': row.get('engagement_strategy.relationship_building', 'moderate'),
            'documentation_support': row.get('engagement_strategy.documentation_support', 'moderate'),
            'follow_up_frequency': row.get('engagement_strategy.follow_up_frequency', 'regular'),
        },
    }
    return customer

class EnhancedTOCAnalyzer:
    def __init__(self, constraint_categories: Optional[Dict]=None):
        # Use externally provided or default categories
        self.constraint_categories = constraint_categories or CONSTRAINT_CATEGORIES
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)

    # -- insights extraction refactored for DRY --
    def extract_customer_insights(self, customer_data: List[Dict[str, Any]]) -> Dict[str, float]:
        if not customer_data:
            return {}

        total = len(customer_data)
        def ratio_count(condition):
            return sum(1 for c in customer_data if condition(c)) / total

        insights = {
            'channel_preference_digital_ratio': ratio_count(lambda c: c['predicted_behaviors']['channel_preference'] in ['digital', 'phone', 'app']),
            'rural_customer_ratio': ratio_count(lambda c: c['segment_profile']['primary_location_type'] in ['rural_remote', 'rural_accessible']),
            'low_digital_literacy_ratio': ratio_count(lambda c: c['segment_profile']['digital_literacy_level'] < 0.3),
            'high_trust_requirement_ratio': ratio_count(lambda c: c['predicted_behaviors']['trust_requirement'] == 'high'),
            'high_doc_support_ratio': ratio_count(lambda c: c['engagement_strategy']['documentation_support'] == 'high'),
            'relationship_building_essential_ratio': ratio_count(lambda c: c['engagement_strategy']['relationship_building'] == 'essential'),
            'field_visit_preference_ratio': ratio_count(lambda c: c['engagement_strategy']['primary_channel'] == 'field_visit'),
            'product_openness_score': np.mean([
                0.8 if c['predicted_behaviors']['product_openness'] == 'high'
                else 0.5 if c['predicted_behaviors']['product_openness'] == 'moderate'
                else 0.2
                for c in customer_data
            ]) or 0.5,
            'avg_segment_confidence': np.mean([c['segment_confidence'] for c in customer_data]) or 0.7
        }
        # Alias for clarity
        insights['digital_channel_preference_ratio'] = insights['channel_preference_digital_ratio']
        return insights

    def create_constraint_dataframe(self, regional_data: Dict[str, Any], customer_insights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        flat_data = {}
        for section in ['behavioral_aggregates', 'infrastructure_metrics', 'service_metrics', 'market_context']:
            flat_data.update(regional_data.get(section, {}))
        if customer_insights:
            flat_data.update(customer_insights)
        df = pd.DataFrame([flat_data])
        df['region_id'] = regional_data.get('region_id', 'unknown')
        return df

    def normalize_constraint_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        normalized_df = df.copy()

        lower_is_better = [
            'product_awareness_score', 'past_campaign_response_rate',
            'bank_branches_per_1000', 'agent_availability_score',
            'kyc_completion_rate', 'application_approval_rate',
            'mobile_network_coverage', 'avg_digital_acceptance',
            'road_connectivity_index', 'electricity_availability',
            'product_openness_score', 'avg_segment_confidence',
            'channel_preference_digital_ratio', 'digital_channel_preference_ratio'
        ]
        higher_is_better = [
            'customer_complaint_rate', 'avg_processing_days',
            'local_economic_stress_index', 'avg_price_sensitivity',
            'avg_trust_requirement', 'rural_customer_ratio',
            'low_digital_literacy_ratio', 'high_trust_requirement_ratio',
            'high_doc_support_ratio', 'relationship_building_essential_ratio',
            'field_visit_preference_ratio'
        ]

        for col in df.columns:
            if col == 'region_id':
                continue
            if any(ind in col for ind in lower_is_better):
                normalized_df[f'{col}_constraint'] = 1.0 - df[col].clip(0, 1)
            elif any(ind in col for ind in higher_is_better):
                if 'processing_days' in col:
                    normalized_df[f'{col}_constraint'] = (df[col] / 30.0).clip(0, 1)
                else:
                    normalized_df[f'{col}_constraint'] = df[col].clip(0, 1)
            else:
                normalized_df[f'{col}_constraint'] = 1.0 - df[col].clip(0, 1)

        return normalized_df

    def calculate_constraint_scores(self, normalized_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        constraint_scores = {}

        for category, config in self.constraint_categories.items():
            systemic_cols = [col for indicator in config['indicators'] for col in normalized_df.columns if indicator in col and col.endswith('_constraint')]
            behavioral_cols = [col for signal in config['behavioral_signals'] for col in normalized_df.columns if signal in col and col.endswith('_constraint')]
            customer_cols = [col for signal in config.get('customer_signals', []) for col in normalized_df.columns if signal in col and col.endswith('_constraint')]

            systemic_score = normalized_df[systemic_cols].mean(axis=1).iloc[0] if systemic_cols else 0.0
            behavioral_score = normalized_df[behavioral_cols].mean(axis=1).iloc[0] if behavioral_cols else 0.0
            customer_score = normalized_df[customer_cols].mean(axis=1).iloc[0] if customer_cols else 0.0

            weights = {
                'systemic': 0.5,
                'behavioral': 0.3,
                'customer': 0.2
            }
            components = []
            if systemic_score > 0:
                components.append(('systemic', systemic_score, weights['systemic']))
            if behavioral_score > 0:
                components.append(('behavioral', behavioral_score, weights['behavioral']))
            if customer_score > 0:
                components.append(('customer', customer_score, weights['customer']))

            if components:
                total_weight = sum(weight for _, _, weight in components)
                combined_score = sum(score * (weight / total_weight) for _, score, weight in components)
            else:
                combined_score = 0.0

            constraint_scores[category] = {
                'severity': combined_score,
                'systemic_component': systemic_score,
                'behavioral_component': behavioral_score,
                'customer_component': customer_score,
                'weighted_severity': combined_score * config['weight'],
                'is_primary_candidate': combined_score > config['threshold'],
                'impact_potential': combined_score * config['impact_multiplier']
            }

        return constraint_scores

    def identify_secondary_constraints(self, constraint_scores: Dict[str,Dict[str,Any]], primary_constraint: str) -> List[Dict[str,Any]]:
        secondary_constraints = []

        sorted_scores = sorted(
            [(name, data) for name, data in constraint_scores.items() if name != primary_constraint],
            key=lambda x: x[1]['weighted_severity'],
            reverse=True
        )
        for name, data in sorted_scores[:2]:
            if data['severity'] > 0.3:  # minimum threshold for secondary
                secondary_constraints.append({
                    'type': name,
                    'severity': data['severity'],
                    'likelihood_of_emergence': min(0.9, data['severity'] * 1.2)
                })

        return secondary_constraints

    def analyze_dataset(self, csv_file_path: str, dataset_name: str = "Dataset") -> Optional[Dict[str,Any]]:
        logger.info(f"Analyzing {dataset_name} ({csv_file_path})...")

        customer_data = load_customer_data_from_csv(csv_file_path)
        if not customer_data:
            logger.warning(f"No valid data found in {csv_file_path}")
            return None

        regional_data = generate_synthetic_regional_data(f"{dataset_name.upper()}_REGION")
        customer_insights = self.extract_customer_insights(customer_data)
        df = self.create_constraint_dataframe(regional_data, customer_insights)
        normalized_df = self.normalize_constraint_indicators(df)
        constraint_scores = self.calculate_constraint_scores(normalized_df)

        # Identify primary constraint
        primary_name = max(constraint_scores.items(), key=lambda x: x[1]['weighted_severity'])[0]
        primary_data = constraint_scores[primary_name]
        secondary_constraints = self.identify_secondary_constraints(constraint_scores, primary_name)

        return {
            'dataset_name': dataset_name,
            'csv_file': csv_file_path,
            'customer_count': len(customer_data),
            'regional_data': regional_data,
            'primary_constraint': {
                'type': primary_name,
                'severity': primary_data['severity'],
                'confidence': 0.90,
                'impact_potential': primary_data['impact_potential'],
                'systemic_component': primary_data['systemic_component'],
                'behavioral_component': primary_data['behavioral_component'],
                'customer_component': primary_data.get('customer_component', 0.0)
            },
            'secondary_constraints': secondary_constraints,
            'constraint_scores': {k: {'severity': v['severity'], 'weighted_severity': v['weighted_severity']} for k, v in constraint_scores.items()},
            'customer_insights': customer_insights
        }

    def analyze_multiple_datasets(self, csv_files: List[str]) -> List[Dict[str,Any]]:
        results = []
        for file in csv_files:
            if os.path.exists(file):
                dataset_name = os.path.splitext(os.path.basename(file))[0].replace('_', ' ').title()
                result = self.analyze_dataset(file, dataset_name)
                if result:
                    results.append(result)
            else:
                logger.warning(f"CSV file {file} not found")
        return results

def print_analysis_results(results: List[Dict[str,Any]]):
    print("\n" + "="*80)
    print("MODEL 4: TOC ANALYSIS RESULTS FOR MULTIPLE DATASETS")
    print("="*80)

    for i, result in enumerate(results, start=1):
        print(f"\n{i}. DATASET: {result['dataset_name']}")
        print(f"   File: {result['csv_file']}")
        print(f"   Customers: {result['customer_count']}")
        pc = result['primary_constraint']
        print(f"   Primary Constraint: {pc['type']} ({pc['severity']:.1%} severity)")
        print(f"   Confidence: {pc['confidence']:.1%}")
        if result['secondary_constraints']:
            print("   Secondary Constraints:")
            for sc in result['secondary_constraints']:
                print(f"     - {sc['type']}: {sc['severity']:.1%}")
        print("   Regional Context:")
        regional = result['regional_data']
        print(f"     - Trust Requirement: {regional['behavioral_aggregates']['avg_trust_requirement']:.1%}")
        print(f"     - Digital Acceptance: {regional['behavioral_aggregates']['avg_digital_acceptance']:.1%}")
        print(f"     - Product Awareness: {regional['market_context']['product_awareness_score']:.1%}")

def save_results_for_model2(results: List[Dict[str,Any]], output_file: str = 'model4_results_for_model2.json') -> str:
    model2_ready = [{
        'dataset_name': r['dataset_name'],
        'primary_constraint': r['primary_constraint'],
        'secondary_constraints': r['secondary_constraints'],
        'customer_insights': r['customer_insights']
    } for r in results]

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(model2_ready, f, indent=2)
    logger.info(f"Results saved to: {output_file}")
    return output_file

def save_results_separately(results: List[Dict[str,Any]], base_output_name="model4_results") -> List[str]:
    """Save results grouped by dataset_name into separate JSON files."""
    output_files = []
    grouped = {}
    for r in results:
        grouped.setdefault(r['dataset_name'], []).append(r)

    for dataset_name, dataset_results in grouped.items():
        filename = f"{base_output_name}_{dataset_name.replace(' ', '_').lower()}.json"
        model2_ready = [{
            'dataset_name': r['dataset_name'],
            'primary_constraint': r['primary_constraint'],
            'secondary_constraints': r['secondary_constraints'],
            'customer_insights': r['customer_insights']
        } for r in dataset_results]

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(model2_ready, f, indent=2)
        logger.info(f"Results for '{dataset_name}' saved to: {filename}")
        output_files.append(filename)

    return output_files

def main():
    logger.info("MODEL 4: MULTI-CSV TOC ANALYZER")

    analyzer = EnhancedTOCAnalyzer()

    csv_files = [
        'customer_segmentation_output.csv',
        'new_customer_segmentation_output.csv'
    ]

    results = analyzer.analyze_multiple_datasets(csv_files)
    print_analysis_results(results)

    output_files = save_results_separately(results)
    logger.info(f"Saved results into {len(output_files)} separate files: {output_files}")
    return results, output_files


if __name__ == "__main__":
    try:
        logger.info("Starting Model 4 analysis...")
        results, model2_file = main()
        logger.info("Model 4 analysis completed successfully!")
    except Exception as e:
        logger.error(f"Error in Model 4 analysis: {e}", exc_info=True)
        raise
