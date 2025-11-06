import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
import pprint
import numpy as np
import pandas as pd
import joblib

def calculate_proxy_confidence(kproto, data_matrix, clusters, cat_indices):
    n_samples = data_matrix.shape[0]
    n_clusters = kproto.n_clusters
    confidences = np.zeros(n_samples)

    centroids_num = kproto.cluster_centroids_[0]
    centroids_cat = kproto.cluster_centroids_[1]

    # Determine numeric feature count dynamically
    if centroids_num.ndim == 1:
        num_features = 1
    else:
        num_features = centroids_num.shape[1]

    for i in range(n_samples):
        sample = data_matrix[i]
        assigned_cluster = clusters[i]

        distances = []
        for c in range(n_clusters):
            num_dist = np.linalg.norm(sample[:num_features] - centroids_num[c])

            cat_dist = 0
            cat_centroid = centroids_cat[c]
            for idx, cat_idx in enumerate(cat_indices):
                if centroids_cat.ndim == 1:
                    cat_value = cat_centroid
                else:
                    cat_value = cat_centroid[idx]

                if sample[num_features + idx] != cat_value:
                    cat_dist += 1

            total_dist = num_dist + cat_dist
            distances.append(total_dist)


        distances = np.array(distances)
        assigned_dist = distances[assigned_cluster]

        conf = 1 - assigned_dist / (distances.sum() + 1e-6)
        confidences[i] = np.clip(conf, 0, 1)

    return confidences

# Load full data
data_full = pd.read_csv("mod1data.csv")

# Sample 5,000 rows (if total rows > 5000), otherwise use all rows
sample_size = 100 if data_full.shape[0] > 100 else data_full.shape[0]
np.random.seed(42)
sample_indices = np.random.choice(data_full.index, size=sample_size, replace=False)

# Create sampled data for all subsequent steps
data = data_full.loc[sample_indices].reset_index(drop=True)

# Map columns as before
data['district'] = data['city']
data['distance_to_post_office'] = data['number_of_post_offices_per_sqkm']
data['distance_to_bank'] = data['number_of_banks_per_sqkm']
# Select required columns
required_columns = [
  'customer_id','age','gender','marital_status','family_size','dependents','education_level',
  'monthly_income','income_sources','income_stability','existing_bank_accounts','existing_loans',
  'existing_insurance','state','district','location_type','distance_to_post_office',
  'distance_to_bank','connectivity_score','occupation_category','land_ownership','land_type',
  'crop_types','seasonal_income_variance','digital_literacy','smartphone_usage',
  'risk_appetite','financial_literacy_score','language_preference'
]
data = data[required_columns]


# -------------------------------------
# Step 2: Data Preprocessing
# -------------------------------------

def preprocess_data(data):
    """Enhanced preprocessing for mixed data types"""
    
    # Handle missing values
    data = data.ffill().bfill()
    
    # Identify column types more intelligently
    categorical_cols = []
    numeric_cols = []
    
    for col in data.columns:
        if data[col].dtype == 'object' or col in ['education_level']:
            categorical_cols.append(col)
        elif pd.api.types.is_numeric_dtype(data[col]) and col != 'customer_id':
            numeric_cols.append(col)
    
    # Remove customer_id from processing
    if 'customer_id' in categorical_cols:
        categorical_cols.remove('customer_id')
    
    # Encode categorical variables
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col + '_encoded'] = le.fit_transform(data[col].astype(str))
        le_dict[col] = le
        
    # Update column lists
    categorical_encoded_cols = [col + '_encoded' for col in categorical_cols]
    
    # Scale numeric features
    scaler = MinMaxScaler()
    data_scaled = data.copy()
    if numeric_cols:
        data_scaled[numeric_cols] = scaler.fit_transform(data_scaled[numeric_cols])
    
    # Prepare for clustering
    cluster_cols = numeric_cols + categorical_encoded_cols
    cat_indices = [cluster_cols.index(col) for col in categorical_encoded_cols]
    
    return data_scaled, cluster_cols, cat_indices, le_dict, scaler

# -------------------------------------
# Step 3: Enhanced Clustering with Optimal K
# -------------------------------------

def find_optimal_clusters(data_matrix, cat_indices, max_k=12):
    """Find optimal number of clusters using elbow method"""
    
    costs = []
    K_range = range(3, 9)
    
    for k in K_range:
        try:
            kproto = KPrototypes(n_clusters=k, init='Cao', random_state=42, verbose=0, n_jobs=1)
            kproto.fit(data_matrix, categorical=cat_indices)
            costs.append(kproto.cost_)
        except:
            costs.append(float('inf'))
    
    # Find elbow point
    if len(costs) > 2:
        kl = KneeLocator(K_range, costs, curve="convex", direction="decreasing")
        optimal_k = kl.knee if kl.knee else K_range[len(K_range)//2]
    else:
        optimal_k = K_range[0]
    
    optimal_k=8
    return optimal_k, costs, K_range

# -------------------------------------
# Step 4: Segment Labeling
# -------------------------------------

def create_segment_labels(data, clusters):
    """Create meaningful segment names based on cluster characteristics"""
    
    segment_profiles = {}
    
    for cluster_id in range(max(clusters) + 1):
        cluster_data = data[data['Cluster'] == cluster_id]
        
        # Analyze cluster characteristics
        avg_age = cluster_data['age'].mean()
        avg_income = cluster_data['monthly_income'].mean()
        most_common_location = cluster_data['location_type'].mode().iloc[0] if not cluster_data['location_type'].mode().empty else 'unknown'
        most_common_occupation = cluster_data['occupation_category'].mode().iloc[0] if not cluster_data['occupation_category'].mode().empty else 'unknown'
        avg_digital_literacy = cluster_data['digital_literacy'].mean() if 'digital_literacy' in cluster_data.columns else 0.5
        
        # Create segment label
        age_group = "Young" if avg_age < 30 else "Middle-aged" if avg_age < 50 else "Senior"
        income_group = "High-income" if avg_income > 50000 else "Middle-income" if avg_income > 25000 else "Low-income"
        tech_level = "Digital-savvy" if avg_digital_literacy > 0.7 else "Traditional"
        
        segment_name = f"{age_group} {most_common_occupation.title()} - {most_common_location.replace('_', ' ').title()}"
        
        segment_profiles[cluster_id] = {
            'segment_name': segment_name,
            'avg_age': round(avg_age, 1),
            'avg_income': round(avg_income, 0),
            'location_type': most_common_location,
            'occupation': most_common_occupation,
            'digital_literacy': round(avg_digital_literacy, 2),
            'size': len(cluster_data),
            'income_group': income_group,
            'tech_level': tech_level
        }
    
    return segment_profiles

# -------------------------------------
# Main Execution
# -------------------------------------

def classify_trait_vs_segment(individual_value, segment_avg, tolerance=0.15):
    """
    Classify individual's trait relative to segment average into
    'below', 'typical', 'above' based on +/- tolerance ratio.
    """
    if abs(individual_value - segment_avg) / max(segment_avg, 1e-9) <= tolerance:
        return 'typical'
    elif individual_value > segment_avg:
        return 'above'
    else:
        return 'below'
    
def map_risk_appetite(value):
    # Map risk appetite strings to ordered scale for comparison
    mapping = {'low': 0, 'moderate': 1, 'high': 2}
    return mapping.get(value, 1)  # default moderate


def map_channel_preference(digital_literacy):
    # Rough heuristic for channel preference based on digital literacy
    if digital_literacy > 0.7:
        return 'digital'
    elif digital_literacy > 0.3:
        return 'phone'
    else:
        return 'in_person'

def run_comprehensive_segmentation():
    data_processed, cluster_cols, cat_indices, le_dict, scaler = preprocess_data(data)
    clustering_data = data_processed[cluster_cols].values

    # No sampling here — use full 'data' that is already sampled

    # Find optimal K on full sampled data
    optimal_k, costs, K_range = find_optimal_clusters(clustering_data, cat_indices, max_k=8)

    kproto = KPrototypes(n_clusters=optimal_k, init='Cao', verbose=0, random_state=42)
    clusters = kproto.fit_predict(clustering_data, categorical=cat_indices)
    proxy_confidences = calculate_proxy_confidence(kproto, clustering_data, clusters, cat_indices)
    data_processed['Cluster'] = clusters
    data['Cluster'] = clusters
    segment_profiles = create_segment_labels(data, data['Cluster'])

    # For each customer, construct the detailed output dictionary
    customer_outputs = []

    for idx, row in data.iterrows():
        cluster_id = row['Cluster']
        profile = segment_profiles[cluster_id]

        # Segment confidence placeholder (optional: can use cluster assignment probability)
        segment_confidence = float(proxy_confidences[idx])

        # Individual vs segment comparisons
        age_cmp = classify_trait_vs_segment(row['age'], profile['avg_age'])
        income_cmp = classify_trait_vs_segment(row['monthly_income'], profile['avg_income'])
        digital_cmp = classify_trait_vs_segment(row['digital_literacy'], profile['digital_literacy'])

        risk_cmp = 'typical'
        # If risk appetite exists in both individual and segment profile, compare
        try:
            individual_risk = map_risk_appetite(row['risk_appetite'])
            segment_risk = map_risk_appetite(profile.get('risk_appetite', 'moderate'))
            diff = individual_risk - segment_risk
            if diff > 0:
                risk_cmp = 'above'
            elif diff < 0:
                risk_cmp = 'below'
            else:
                risk_cmp = 'typical'
        except:
            risk_cmp = 'typical'

        # Basic predicted behaviors heuristics
        # Product openness: high if income and digital literacy are above avg, else moderate or low
        prod_openness = 'moderate'
        if income_cmp == 'above' and digital_cmp in ['typical', 'above']:
            prod_openness = 'high'
        elif income_cmp == 'below' and digital_cmp == 'below':
            prod_openness = 'low'

        channel_preference = map_channel_preference(row['digital_literacy'])

        # Decision speed heuristic
        decision_speed = 'moderate'
        if row['age'] < 30:
            decision_speed = 'fast'
        elif row['age'] > 60:
            decision_speed = 'slow'

        # Price sensitivity by income level
        price_sensitivity = 'moderate'
        if income_cmp == 'below':
            price_sensitivity = 'high'
        elif income_cmp == 'above':
            price_sensitivity = 'low'

        # Trust requirement high if digital literacy low or risk appetite low
        trust_requirement = 'high' if row['digital_literacy'] < 0.3 or row['risk_appetite'] == 'low' else 'moderate'

        # Engagement strategy heuristic
        primary_channel = 'field_visit' if channel_preference == 'in_person' else channel_preference
        message_complexity = 'simple' if row['education_level'] < 2 else 'moderate'
        relationship_building = 'essential' if trust_requirement == 'high' else 'moderate'
        documentation_support = 'high' if profile['income_group'] == 'Low-income' else 'moderate'
        follow_up_frequency = 'regular' if trust_requirement == 'high' else 'minimal'

        customer_segment_output = {
            'customer_id': row['customer_id'],
            'cluster_id': int(cluster_id),
            'segment_name': profile['segment_name'],
            'segment_confidence': segment_confidence,
            'segment_profile': {
                'avg_age': profile['avg_age'],
                'avg_income': profile['avg_income'],
                'primary_location_type': profile['location_type'],
                'primary_occupation': profile['occupation'],
                'digital_literacy_level': profile['digital_literacy'],
                'risk_appetite': row.get('risk_appetite', 'moderate'),
                'income_stability': row.get('income_stability', 0),
                'financial_literacy': row.get('financial_literacy_score', 0)
            },
            'individual_traits': {
                'age_vs_segment': age_cmp,
                'income_vs_segment': income_cmp,
                'digital_vs_segment': digital_cmp,
                'risk_vs_segment': risk_cmp
            },
            'predicted_behaviors': {
                'product_openness': prod_openness,
                'channel_preference': channel_preference,
                'decision_speed': decision_speed,
                'price_sensitivity': price_sensitivity,
                'trust_requirement': trust_requirement
            },
            'engagement_strategy': {
                'primary_channel': primary_channel,
                'message_complexity': message_complexity,
                'relationship_building': relationship_building,
                'documentation_support': documentation_support,
                'follow_up_frequency': follow_up_frequency
            }
        }

        customer_outputs.append(customer_segment_output)

        joblib.dump(kproto, 'kproto_model.pkl')
        joblib.dump(le_dict, 'label_encoders.pkl')
        joblib.dump(scaler, 'scaler.pkl')

    return customer_outputs, segment_profiles


# Run and print example outputs
outputs, segment_profiles = run_comprehensive_segmentation()

def preprocess_new_data(new_data, le_dict, scaler, categorical_cols, numeric_cols):
    # Map columns if necessary (same as for training)

    # Encode categorical columns using saved LabelEncoders
    for col in categorical_cols:
        le = le_dict[col]
        new_data[col + '_encoded'] = new_data[col].map(lambda x: le.transform([str(x)])[0] if x in le.classes_ else -1)  # Handle unseen categories as -1

    # Scale numeric columns using saved scaler
    new_data_scaled = new_data.copy()
    if numeric_cols:
        new_data_scaled[numeric_cols] = scaler.transform(new_data_scaled[numeric_cols])

    # Prepare matrix for clustering
    cluster_cols = numeric_cols + [col + '_encoded' for col in categorical_cols]
    data_matrix = new_data_scaled[cluster_cols].values

    # Identify cat_indices
    cat_indices = [cluster_cols.index(col + '_encoded') for col in categorical_cols]

    return new_data_scaled, data_matrix, cat_indices

def predict_new_data(new_data, kproto, le_dict, scaler):
    # categorical_cols and numeric_cols as before, matching original data columns
    categorical_cols = list(le_dict.keys())
    numeric_cols = [col for col in new_data.columns if pd.api.types.is_numeric_dtype(new_data[col]) and col not in categorical_cols and col != 'customer_id']

    # Preprocess new data to get clustering matrix
    new_data_scaled, data_matrix, cat_indices = preprocess_new_data(new_data, le_dict, scaler, categorical_cols, numeric_cols)

    # Predict clusters on preprocessed data
    clusters = kproto.predict(data_matrix, categorical=cat_indices)

    # Calculate confidence
    proxy_confidences = calculate_proxy_confidence(kproto, data_matrix, clusters, cat_indices)

    # Attach clustering results to original raw data (not scaled)
    new_data = new_data.copy()
    new_data['Cluster'] = clusters
    new_data['Confidence'] = proxy_confidences

    # Optionally, add segment labels etc. here using trained segment profiles

    return new_data  # Return raw data + cluster columns


# Load saved models
kproto = joblib.load('kproto_model.pkl')
le_dict = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')

# Load new input
new_data = pd.read_csv('new_customer_data.csv')

# Map and filter columns same as training
new_data['district'] = new_data['city']
new_data['distance_to_post_office'] = new_data['number_of_post_offices_per_sqkm']
new_data['distance_to_bank'] = new_data['number_of_banks_per_sqkm']

required_columns = ['customer_id','age','gender','marital_status','family_size','dependents','education_level',
  'monthly_income','income_sources','income_stability','existing_bank_accounts','existing_loans',
  'existing_insurance','state','district','location_type','distance_to_post_office',
  'distance_to_bank','connectivity_score','occupation_category','land_ownership','land_type',
  'crop_types','seasonal_income_variance','digital_literacy','smartphone_usage',
  'risk_appetite','financial_literacy_score','language_preference']
new_data = new_data[required_columns]

# Predict clusters and generate outputs
results = predict_new_data(new_data, kproto, le_dict, scaler)

# Save to CSV
results.to_csv('new_customer_segmentation_output.csv', index=False)


def flatten_customer_output(cust_out):
    flat_dict = {}
    flat_dict['customer_id'] = cust_out['customer_id']
    flat_dict['cluster_id'] = cust_out['cluster_id']
    flat_dict['segment_name'] = cust_out['segment_name']
    flat_dict['segment_confidence'] = cust_out['segment_confidence']
    
    # Flatten segment_profile
    for key, val in cust_out['segment_profile'].items():
        flat_dict[f'segment_profile.{key}'] = val
        
    # Flatten individual_traits
    for key, val in cust_out['individual_traits'].items():
        flat_dict[f'individual_traits.{key}'] = val
        
    # Flatten predicted_behaviors
    for key, val in cust_out['predicted_behaviors'].items():
        flat_dict[f'predicted_behaviors.{key}'] = val
        
    # Flatten engagement_strategy
    for key, val in cust_out['engagement_strategy'].items():
        flat_dict[f'engagement_strategy.{key}'] = val
        
    return flat_dict

# Flatten all outputs
flattened_outputs = [flatten_customer_output(cust) for cust in outputs]

# Create DataFrame
df_output = pd.DataFrame(flattened_outputs)

# Save to CSV
df_output.to_csv('customer_segmentation_output.csv', index=False)


pp = pprint.PrettyPrinter(indent=2, compact=False, width=120)

for cust_out in outputs:
    # Convert numpy numeric values to native Python types for cleaner printing
    cust_out['cluster_id'] = int(cust_out['cluster_id'])
    cust_out['segment_confidence'] = float(cust_out['segment_confidence'])

    for k in ['avg_age', 'avg_income', 'digital_literacy_level', 'income_stability', 'financial_literacy']:
        if k in cust_out['segment_profile']:
            cust_out['segment_profile'][k] = float(cust_out['segment_profile'][k])
    
    # Print header comment like your example
    print('customer_segment_output = ', end='')
    
    # Pretty-print the dictionary to stdout
    pp.pprint(cust_out)
    print('\n')
