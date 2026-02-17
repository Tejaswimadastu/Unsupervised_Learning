import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# -------------------- APP TITLE --------------------
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("🟢 Customer Segmentation Dashboard")
st.write("This system uses K-Means Clustering to group customers based on their purchasing behavior and similarities.")

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")

df = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# -------------------- SIDEBAR INPUTS --------------------
st.sidebar.header("⚙️ Clustering Controls")

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

feature1 = st.sidebar.selectbox("Select Feature 1", numeric_cols)
feature2 = st.sidebar.selectbox("Select Feature 2", numeric_cols)

k = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=10, value=5)

random_state = st.sidebar.number_input("Random State (Optional)", value=42)

run = st.sidebar.button("🟦 Run Clustering")

# -------------------- RUN CLUSTERING --------------------
if run:
    if feature1 == feature2:
        st.error("Please select two different features!")
    else:
        X = df[[feature1, feature2]]

        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # KMeans
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        df["Cluster"] = kmeans.fit_predict(X_scaled)

        # -------------------- VISUALIZATION --------------------
        st.subheader("📈 Cluster Visualization")

        fig, ax = plt.subplots()
        scatter = ax.scatter(X.iloc[:,0], X.iloc[:,1], c=df["Cluster"])
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        ax.scatter(centers[:,0], centers[:,1], s=300, marker="X")

        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        ax.set_title("Customer Clusters")
        st.pyplot(fig)

        # -------------------- CLUSTER SUMMARY --------------------
        # -------------------- CLUSTER SUMMARY --------------------
        st.subheader("📋 Cluster Summary Table")

        df["Cluster"] = kmeans.fit_predict(X_scaled)   # Ensure Cluster exists

        summary = df.groupby("Cluster").agg(
            Count=("Cluster", "count"),
            Avg_Feature1=(feature1, "mean"),
            Avg_Feature2=(feature2, "mean")
)

        st.dataframe(summary)


        # -------------------- BUSINESS INTERPRETATION --------------------
        st.subheader("💼 Business Interpretation")

        for c in summary.index:
            avg1 = summary.loc[c, "Avg_Feature1"]
            avg2 = summary.loc[c, "Avg_Feature2"]

            if avg2 > X[feature2].mean():
                st.success(f"Cluster {c}: High-spending customers across selected categories.")
            elif avg2 < X[feature2].mean():
                st.warning(f"Cluster {c}: Budget-conscious customers with lower spending.")
            else:
                st.info(f"Cluster {c}: Moderate spenders with selective purchasing behavior.")

        # -------------------- USER GUIDANCE --------------------
        st.info("📌 Customers in the same cluster exhibit similar purchasing behaviour and can be targeted with similar business strategies.")
