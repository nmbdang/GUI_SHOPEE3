import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import pickle
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# ---- Load dữ liệu ----
@st.cache_data
def load_data():
    df_products = pd.read_csv('San_pham_temp.csv')
    df_ratings = pd.read_csv('user_ratings.csv')

    with open('nn_model.pkl', 'rb') as f:
        nn_model = pickle.load(f)

    with open('tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)

    with open('surprise_model.pkl', 'rb') as f:
        surprise_model = pickle.load(f)

    product_id_to_name = pd.Series(df_products.ten_san_pham.values, index=df_products.ma_san_pham).to_dict()

    return df_products, df_ratings, nn_model, tfidf_matrix, surprise_model, product_id_to_name

# ---- Gợi ý theo nội dung ----
def get_content_recommendations(product_code, df_products, _tfidf_matrix):
    idx_list = df_products.index[df_products['ma_san_pham'] == product_code].tolist()
    if not idx_list:
        return pd.DataFrame()
    idx = idx_list[0]
    product_vector = _tfidf_matrix[idx]
    cos_sim = cosine_similarity(product_vector, _tfidf_matrix).flatten()
    sim_scores = sorted(enumerate(cos_sim), key=lambda x: x[1], reverse=True)[1:7]
    product_indices = [i[0] for i in sim_scores]
    return df_products.iloc[product_indices]

# ---- Gợi ý theo người dùng ----
def get_top_n_recommendations(user_id, model, df_ratings, df_products, n=6):
    rated_items = df_ratings[df_ratings['user_id'] == user_id]['product_id'].unique()
    all_items = df_products['ma_san_pham'].unique()
    items_to_predict = [item for item in all_items if item not in rated_items]

    predictions = []
    for item_id in items_to_predict:
        pred = model.predict(user_id, item_id)
        predictions.append((item_id, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n_items = [item_id for item_id, _ in predictions[:n]]

    return df_products[df_products['ma_san_pham'].isin(top_n_items)]

# ---- Hiển thị sản phẩm dạng thẻ ----
def display_product_card(product):
    st.markdown(
        f"""
        <div style="border: 1px solid #e6e6e6; border-radius: 16px; padding: 16px; margin-bottom: 16px; 
                    box-shadow: 0 4px 12px rgba(0,0,0,0.08); background-color: #fff;">
            <h4 style="color: #ee4d2d;">🛒 {product['ten_san_pham']}</h4>
            <p><b>🔖 Mã sản phẩm:</b> <code>{product['ma_san_pham']}</code></p>
            <details>
                <summary style='cursor: pointer;'>📄 <b>Xem mô tả</b></summary>
                <p style='margin-top:10px;'>{product.get('mo_ta', 'Không có mô tả.')}</p>
            </details>
            <div style="text-align:center; margin-top:10px;">
                <a href="#" target="_blank" 
                   style="background-color:#ee4d2d; color:white; padding:10px 20px; 
                          border-radius:8px; text-decoration:none; font-weight:bold;">
                   🛍️ Mua ngay
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---- Giao diện Streamlit ----
def main():
    st.set_page_config(layout="wide", page_title="🛒 Hệ thống gợi ý Shopee")

    banner = Image.open('shopee-banner-shopee-1.png')
    st.image(banner, use_column_width=True)

    st.title("🛍️ Hệ thống Gợi Ý Sản Phẩm Shopee")

    df_products, df_ratings, nn_model, tfidf_matrix, surprise_model, product_id_to_name = load_data()

    st.sidebar.title("⚙️ Tuỳ chọn")
    choice = st.sidebar.selectbox("Chọn phương pháp gợi ý:", 
                                   ["🔎 Gợi ý theo nội dung", "👤 Gợi ý theo người dùng"])

    if choice == "🔎 Gợi ý theo nội dung":
        st.header("🔎 Gợi ý sản phẩm tương tự")

        product_options = [(row['ten_san_pham'], row['ma_san_pham']) for _, row in df_products.iterrows()]
        selected_product = st.selectbox("📦 Chọn sản phẩm bạn quan tâm:", product_options)

        if selected_product:
            selected_product_name, selected_product_code = selected_product
            selected_info = df_products[df_products['ma_san_pham'] == selected_product_code].iloc[0]

            st.success(f"📌 Bạn đã chọn: **{selected_product_name}**")

            st.markdown("##### 📝 Mô tả sản phẩm:")
            st.write(selected_info.get('mo_ta', 'Không có mô tả.'))

            st.subheader("✨ Gợi ý sản phẩm tương tự:")

            recommended_products = get_content_recommendations(selected_product_code, df_products, tfidf_matrix)

            if not recommended_products.empty:
                cols = st.columns(3)
                for idx, (_, row) in enumerate(recommended_products.iterrows()):
                    with cols[idx % 3]:
                        display_product_card(row)
            else:
                st.warning("Không tìm thấy sản phẩm tương tự!")

    elif choice == "👤 Gợi ý theo người dùng":
        st.header("👤 Gợi ý sản phẩm cho người dùng")

        user_ids = df_ratings['user_id'].unique().tolist()
        selected_user = st.selectbox("👤 Chọn User ID:", user_ids)

        if selected_user:
            st.success(f"📌 Đang gợi ý cho User ID: **{selected_user}**")

            st.subheader("✨ Sản phẩm bạn có thể thích:")

            recommended_products = get_top_n_recommendations(selected_user, surprise_model, df_ratings, df_products)

            if not recommended_products.empty:
                cols = st.columns(3)
                for idx, (_, row) in enumerate(recommended_products.iterrows()):
                    with cols[idx % 3]:
                        display_product_card(row)
            else:
                st.warning("Không tìm thấy sản phẩm phù hợp!")

if __name__ == "__main__":
    main()

