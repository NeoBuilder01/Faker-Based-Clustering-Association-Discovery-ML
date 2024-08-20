import numpy as np
import pandas as pd
from faker import Faker
import random
import plotly.express as px
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore', category=UserWarning, message='Could not find the number of physical cores')

np.random.seed(32)
random.seed(32)

fake = Faker()

def generate_data(num_products=10, num_customers=100, num_transactions=500):
    products = [fake.word() for _ in range(num_products)]
    
    transactions = []

    for _ in range(num_transactions):
        customer_id = random.randint(1, num_customers)
        basket_size = random.randint(1, 5)
        basket = random.sample(products, basket_size)
        transactions.append({'customer_id': customer_id, 'products': basket})

    df = pd.DataFrame(transactions)
    df_encoded = df.explode('products').pivot_table(
        index='customer_id',
        columns='products',
        aggfunc=lambda x: 1,
        fill_value=0
    )
    return df_encoded

def simple_apriori(df, min_support=0.1, min_confidence=0.5):
    def support(item_set):
        return (df[list(item_set)].sum(axis=1) == len(item_set)).mean()
    
    items = set(df.columns)
    rules = []

    for k in range(2, len(items) + 1):
        item_sets = [frozenset(item) for item in combinations(items, k) if support(item) >= min_support]

        for item_set in item_sets:
            item_set = frozenset(item_set)
            for i in range(1, len(item_set)):
                for antecedents in combinations(item_set, i):
                    antecedents = frozenset(antecedents)
                    consequents = item_set - antecedents
                    conf = support(item_set) / support(consequents)
                    if conf >= min_confidence:
                        lift = conf / (support(consequents) or 1)  # Avoid division by zero
                        rules.append({
                            'antecedents': ', '.join(antecedents),
                            'consequents': ', '.join(consequents),
                            'support': support(item_set),
                            'confidence': conf,
                            'lift': lift
                        })
                        if len(rules) >= 10:  # stop if we have 10 rules
                            break
            if len(rules) >= 10:
                break
                
    return pd.DataFrame(rules).sort_values('lift', ascending=False)

def perform_kmeans_with_progress(df, n_clusters=3, max_iterations=2):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    kmeans = KMeans(n_clusters=n_clusters, random_state=32, max_iter=100)
    
    iteration_count = 0
    while iteration_count < max_iterations:
        kmeans.fit(df_scaled)
        iteration_count += 1
        # Return the cluster labels for the current iteration
        yield kmeans.labels_

    # Return the final cluster labels after the loop
    yield kmeans.labels_

def visualize_apriori_rules(rules, top_n=10):
    top_rules = rules.head(top_n)

    fig = px.scatter_3d(
        top_rules, x='support', y='confidence', z='lift',
        color='lift', size='support',
        hover_name='antecedents', hover_data=['consequents'],
        labels={'support': 'support', 'confidence': 'confidence', 'lift': 'lift'},
        title=f"Top {top_n} Association Rules"
    )
    
    return fig

def visualize_kmeans_clusters(df, cluster_labels):
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df)

    fig = px.scatter_3d(
        x=pca_result[:, 0], y=pca_result[:, 1], z=pca_result[:, 2],
        color=cluster_labels,
        labels={'x': 'PCA 1', 'y': 'PCA 2', 'z': 'PCA 3'},
        title='Customer Clusters'
    )
    
    return fig

def main():
    print("Gathering synthetic data..........")

    df_encoded = generate_data(num_products=10, num_customers=100, num_transactions=500)
    print("Data gathering complete")
    print(f"Dataset shape: {df_encoded.shape}")

    print("Performing Apriori Algorithm.....")
    rules = simple_apriori(df_encoded, min_support=0.1, min_confidence=0.5)

    if not rules.empty:
        print(f"Apriori algorithm complete. Found {len(rules)} rules.")
        viz = visualize_apriori_rules(rules)
        viz.write_html("apriori3d.html")
        print("Apriori rules visual saved as 'apriori3d.html'.")
    else:
        print("Apriori algorithm failed to find any rules.")

    print("Performing KMeans.")

    kmeans_generator = perform_kmeans_with_progress(df_encoded, n_clusters=3, max_iterations=2)
    
    # Save only the first two intermediate visualizations
    for i, labels in enumerate(kmeans_generator):
        if i < 2:  # Only save the first two intermediate visualizations
            print(f"KMeans iteration {i}")
            viz = visualize_kmeans_clusters(df_encoded, labels)
            viz.write_html(f"customer_cluster_3d_step_{i}.html")
            print(f"Intermediate visual saved as customer_cluster_3d_step_{i}.html")

    # Generate final visualization
    final_labels = labels  # Last generated labels
    print("KMeans clustering complete")

    final_viz = visualize_kmeans_clusters(df_encoded, final_labels)
    final_viz.write_html("customer_cluster_3d_final.html")
    print("Final customer cluster visualization saved as 'customer_cluster_3d_final.html'.")
    print("Analysis complete!")

if __name__ == "__main__":
    main()
