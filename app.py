import os
import re

import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from apify_client import ApifyClient

from feature_engineering import calculate_features


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Instagram Profile Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.html("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at 5% 5%,
            rgba(131, 58, 180, 0.20),
            transparent 28%
        ),
        radial-gradient(
            circle at 95% 10%,
            rgba(225, 48, 108, 0.16),
            transparent 30%
        ),
        radial-gradient(
            circle at 50% 100%,
            rgba(253, 29, 29, 0.08),
            transparent 30%
        ),
        #08080d;
    color: #ffffff;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* =====================================================
   HERO
   ===================================================== */

.hero {
    text-align: center;
    padding: 25px 10px 35px 10px;
}

.hero-badge {
    display: inline-block;
    padding: 8px 18px;
    border-radius: 30px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    color: #d7d7e0;
    font-size: 13px;
    letter-spacing: 0.5px;
    margin-bottom: 16px;
}

.hero-title {
    font-size: 48px;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(
        90deg,
        #ffffff,
        #d7a5ff,
        #ff6fae
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #9b9ba8;
    font-size: 17px;
    margin-top: 12px;
}


/* =====================================================
   MAIN CARDS
   ===================================================== */

.card {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 15px 45px rgba(0,0,0,0.25);
}

.card-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 8px;
}

.card-subtitle {
    color: #8f8f9c;
    font-size: 14px;
    line-height: 1.6;
    margin-bottom: 18px;
}


/* =====================================================
   RESULT
   ===================================================== */

.result-card {
    text-align: center;
    padding: 35px 20px;
    margin-top: 30px;
    border-radius: 24px;
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 15px 50px rgba(0,0,0,0.25);
}

.result-label {
    color: #9696a3;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 3px;
}

.result-title {
    font-size: 35px;
    font-weight: 800;
    margin-top: 12px;
}

.real-result {
    color: #55e6a5;
}

.fake-result {
    color: #ff5c70;
}

.confidence {
    font-size: 55px;
    font-weight: 800;
    margin-top: 8px;
}

.confidence-label {
    color: #92929f;
    font-size: 14px;
}


/* =====================================================
   PROBABILITY BARS
   ===================================================== */

.probability-wrapper {
    margin-top: 25px;
    text-align: left;
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
}

.probability-row {
    margin-bottom: 15px;
}

.probability-header {
    display: flex;
    justify-content: space-between;
    color: #c5c5cf;
    font-size: 13px;
    margin-bottom: 7px;
}

.progress-background {
    width: 100%;
    height: 9px;
    background: rgba(255,255,255,0.08);
    border-radius: 20px;
    overflow: hidden;
}

.real-progress {
    height: 100%;
    background: linear-gradient(
        90deg,
        #24c98b,
        #63efb1
    );
    border-radius: 20px;
}

.fake-progress {
    height: 100%;
    background: linear-gradient(
        90deg,
        #e83367,
        #ff6878
    );
    border-radius: 20px;
}


/* =====================================================
   STAT CARDS
   ===================================================== */

.stat-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 17px;
    padding: 20px 10px;
    text-align: center;
}

.stat-value {
    font-size: 25px;
    font-weight: 750;
}

.stat-label {
    color: #8e8e9b;
    font-size: 11px;
    margin-top: 6px;
    letter-spacing: 0.7px;
}


/* =====================================================
   FOOTER
   ===================================================== */

.footer {
    text-align: center;
    color: #686875;
    font-size: 12px;
    margin-top: 45px;
    padding: 18px;
    border-top: 1px solid rgba(255,255,255,0.06);
}


/* =====================================================
   STREAMLIT INPUT
   ===================================================== */

div[data-baseweb="input"] {
    border-radius: 12px;
}


/* =====================================================
   BUTTON
   ===================================================== */

div.stButton > button {
    width: 100%;
    height: 48px;
    border-radius: 12px;
    font-weight: 700;
    border: none;
    color: white;
    background: linear-gradient(
        90deg,
        #833ab4,
        #e1306c,
        #fd1d1d
    );
    transition: 0.2s;
}

div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(225,48,108,0.25);
}


/* =====================================================
   EXPANDER
   ===================================================== */

.streamlit-expanderHeader {
    border-radius: 12px;
}

</style>
""")


# =========================================================
# LOAD ENVIRONMENT
# =========================================================

load_dotenv()

# Read token from .env (local) or Streamlit secrets (cloud deploy)
API_TOKEN = os.getenv("APIFY_API_TOKEN")
if not API_TOKEN:
    try:
        API_TOKEN = st.secrets["APIFY_API_TOKEN"]
    except Exception:
        API_TOKEN = None


# =========================================================
# LOAD MODEL + SCALER
# =========================================================

@st.cache_resource
def load_models():

    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")

    return model, scaler


# =========================================================
# APIFY SCRAPER
# =========================================================

def scrape_profile(username):

    client = ApifyClient(API_TOKEN)

    run_input = {
        "usernames": [username]
    }

    run = client.actor(
        "apify/instagram-profile-scraper"
    ).call(
        run_input=run_input
    )

    items = list(
        client.dataset(
            run.default_dataset_id
        ).iterate_items()
    )

    if not items:
        return None

    return items[0]


# =========================================================
# EXTRACT USERNAME
# =========================================================

def extract_username(url):

    url = url.strip()

    pattern = r"(?:https?://)?(?:www\.)?instagram\.com/([^/?#]+)/?"

    match = re.search(
        pattern,
        url,
        re.IGNORECASE
    )

    if not match:
        return None

    username = match.group(1).strip()

    return username


# =========================================================
# HEADER
# =========================================================

st.html("""
<div class="hero">

    <div class="hero-badge">
        ✦ MACHINE LEARNING • INSTAGRAM ANALYTICS
    </div>

    <div class="hero-title">
        Instagram Profile Detector
    </div>

    <div class="hero-subtitle">
        AI-powered analysis to identify potentially fake Instagram accounts
    </div>

</div>
""")


# =========================================================
# API CHECK
# =========================================================

if not API_TOKEN:

    st.error(
        "Apify API token not found. Please add APIFY_API_TOKEN to your .env file."
    )

    st.stop()


# =========================================================
# INPUT SECTION
# =========================================================

left, right = st.columns(
    [1.15, 0.85],
    gap="large"
)


# =========================================================
# LEFT COLUMN
# =========================================================

with left:

    st.html("""
    <div class="card">

        <div class="card-title">
            🔗 Analyze Instagram Profile
        </div>

        <div class="card-subtitle">
            Enter a public Instagram profile URL and let the
            machine learning model analyze the account.
        </div>

    </div>
    """)

    instagram_url = st.text_input(
        "Instagram Profile URL",
        placeholder="https://www.instagram.com/username/",
        label_visibility="collapsed"
    )

    analyze = st.button(
        "🔍  Analyze Profile"
    )


# =========================================================
# RIGHT COLUMN
# =========================================================

with right:

    st.html("""
    <div class="card">

        <div class="card-title">
            🤖 How It Works
        </div>

        <div class="card-subtitle">
            The complete prediction pipeline works automatically.
        </div>

        <div style="
            line-height: 2;
            color: #c8c8d2;
            font-size: 14px;
        ">

            ① &nbsp; Fetch profile using Apify<br>

            ② &nbsp; Extract profile & behavioral data<br>

            ③ &nbsp; Generate ML features<br>

            ④ &nbsp; Scale the features<br>

            ⑤ &nbsp; Run LightGBM classifier<br>

            ⑥ &nbsp; Generate prediction & confidence

        </div>

    </div>
    """)


# =========================================================
# ANALYSIS
# =========================================================

if analyze:

    # -----------------------------------------------------
    # URL VALIDATION
    # -----------------------------------------------------

    if not instagram_url.strip():

        st.warning(
            "Please enter an Instagram profile URL."
        )

        st.stop()


    username = extract_username(
        instagram_url
    )


    if not username:

        st.error(
            "Invalid Instagram profile URL. Example: https://www.instagram.com/nasa/"
        )

        st.stop()


    # -----------------------------------------------------
    # RUN PIPELINE
    # -----------------------------------------------------

    try:

        model, scaler = load_models()


        # -------------------------------------------------
        # SCRAPE
        # -------------------------------------------------

        with st.spinner(
            f"🔎 Fetching @{username} profile data..."
        ):

            profile = scrape_profile(
                username
            )


        if not profile:

            st.error(
                "Unable to retrieve profile data from Apify."
            )

            st.stop()


        # -------------------------------------------------
        # FEATURE ENGINEERING
        # -------------------------------------------------

        with st.spinner(
            "⚙️ Extracting behavioral features..."
        ):

            features = calculate_features(
                profile
            )


        # -------------------------------------------------
        # SCALE FEATURES
        # -------------------------------------------------

        features_scaled = scaler.transform(
            features
        )


        # Keep feature names
        features_scaled = pd.DataFrame(
            features_scaled,
            columns=features.columns
        )


        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        prediction = model.predict(
            features_scaled
        )

        probabilities = model.predict_proba(
            features_scaled
        )


        real_probability = float(
            probabilities[0][0]
        )

        fake_probability = float(
            probabilities[0][1]
        )


        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        if prediction[0] == 1:

            result = "FAKE ACCOUNT"
            confidence = fake_probability

            result_icon = "🔴"
            result_class = "fake-result"

        else:

            result = "REAL ACCOUNT"
            confidence = real_probability

            result_icon = "🟢"
            result_class = "real-result"


        # =================================================
        # RESULT CARD
        # =================================================

        st.html(f"""
        <div class="result-card">

            <div class="result-label">
                ANALYSIS RESULT
            </div>

            <div class="result-title {result_class}">
                {result_icon} {result}
            </div>

            <div class="confidence">
                {confidence * 100:.2f}%
            </div>

            <div class="confidence-label">
                Prediction Confidence
            </div>

            <div class="probability-wrapper">

                <div class="probability-row">

                    <div class="probability-header">
                        <span>🟢 Real Probability</span>
                        <span>{real_probability * 100:.2f}%</span>
                    </div>

                    <div class="progress-background">

                        <div
                            class="real-progress"
                            style="width:{real_probability * 100:.2f}%"
                        >
                        </div>

                    </div>

                </div>


                <div class="probability-row">

                    <div class="probability-header">
                        <span>🔴 Fake Probability</span>
                        <span>{fake_probability * 100:.2f}%</span>
                    </div>

                    <div class="progress-background">

                        <div
                            class="fake-progress"
                            style="width:{fake_probability * 100:.2f}%"
                        >
                        </div>

                    </div>

                </div>

            </div>

        </div>
        """)


        # =================================================
        # PROFILE OVERVIEW
        # =================================================

        st.markdown("### 📊 Profile Overview")


        followers = profile.get(
            "followersCount",
            0
        ) or 0


        following = profile.get(
            "followsCount",
            0
        ) or 0


        posts = profile.get(
            "postsCount",
            0
        ) or 0


        col1, col2, col3, col4 = st.columns(4)


        # -------------------------------------------------
        # FOLLOWERS
        # -------------------------------------------------

        with col1:

            st.html(f"""
            <div class="stat-card">

                <div class="stat-value">
                    {followers:,}
                </div>

                <div class="stat-label">
                    FOLLOWERS
                </div>

            </div>
            """)


        # -------------------------------------------------
        # FOLLOWING
        # -------------------------------------------------

        with col2:

            st.html(f"""
            <div class="stat-card">

                <div class="stat-value">
                    {following:,}
                </div>

                <div class="stat-label">
                    FOLLOWING
                </div>

            </div>
            """)


        # -------------------------------------------------
        # POSTS
        # -------------------------------------------------

        with col3:

            st.html(f"""
            <div class="stat-card">

                <div class="stat-value">
                    {posts:,}
                </div>

                <div class="stat-label">
                    POSTS
                </div>

            </div>
            """)


        # -------------------------------------------------
        # RISK
        # -------------------------------------------------

        with col4:

            if fake_probability >= 0.70:

                risk = "HIGH"

            elif fake_probability >= 0.40:

                risk = "MEDIUM"

            else:

                risk = "LOW"


            st.html(f"""
            <div class="stat-card">

                <div class="stat-value">
                    {risk}
                </div>

                <div class="stat-label">
                    RISK LEVEL
                </div>

            </div>
            """)


        # =================================================
        # ACCOUNT INFORMATION
        # =================================================

        st.markdown("### 👤 Account Information")


        profile_name = profile.get(
            "fullName",
            username
        ) or username


        profile_picture = (
            "Available"
            if profile.get("profilePicUrl")
            else "Not Available"
        )


        external_link = (
            "Available"
            if profile.get("externalUrl")
            else "Not Available"
        )


        st.html(f"""
        <div class="card">

            <div style="
                color:#b7b7c3;
                line-height:2;
                font-size:14px;
            ">

                <b>Username:</b> @{username}<br>

                <b>Name:</b> {profile_name}<br>

                <b>Profile Picture:</b> {profile_picture}<br>

                <b>External Link:</b> {external_link}<br>

            </div>

        </div>
        """)


        # =================================================
        # MODEL FEATURES
        # =================================================

        with st.expander(
            "🔬 View extracted model features"
        ):

            display_features = pd.DataFrame(
                {
                    "Feature": features.columns,
                    "Value": features.iloc[0].values
                }
            )

            st.dataframe(
                display_features,
                use_container_width=True,
                hide_index=True
            )


    # =====================================================
    # ERROR HANDLING
    # =====================================================

    except Exception as e:

        st.error(
            f"Something went wrong: {str(e)}"
        )


# =========================================================
# FOOTER
# =========================================================

st.html("""
<div class="footer">

    Built with Python • Apify • Scikit-learn • LightGBM • Streamlit

</div>
""")