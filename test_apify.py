import os
import re
import joblib
import pandas as pd

from dotenv import load_dotenv
from apify_client import ApifyClient

from feature_engineering import calculate_features


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

API_TOKEN = os.getenv(
    "APIFY_API_TOKEN"
)

if not API_TOKEN:

    raise ValueError(
        "APIFY_API_TOKEN not found in .env file"
    )


# ============================================================
# 2. LOAD MODEL AND SCALER
# ============================================================

print("\nLoading ML model...")

model = joblib.load(
    "model.pkl"
)

scaler = joblib.load(
    "scaler.pkl"
)

print(
    "Model loaded successfully."
)


# ============================================================
# 3. FEATURE ORDER
# ============================================================

FEATURE_NAMES = [
    "pos",
    "flw",
    "flg",
    "bl",
    "pic",
    "lin",
    "cl",
    "cz",
    "ni",
    "erl",
    "erc",
    "lt",
    "hc",
    "pr",
    "pi"
]


# ============================================================
# 4. GET INSTAGRAM URL
# ============================================================

instagram_url = input(
    "\nEnter Instagram profile URL: "
).strip()


# ============================================================
# 5. EXTRACT USERNAME
# ============================================================

match = re.search(
    r"instagram\.com/([^/?#]+)/?",
    instagram_url
)

if not match:

    raise ValueError(
        "Invalid Instagram profile URL"
    )

username = match.group(1)

print(
    f"\nUsername detected: {username}"
)


# ============================================================
# 6. APIFY CLIENT
# ============================================================

client = ApifyClient(
    API_TOKEN
)

run_input = {
    "usernames": [
        username
    ]
}


# ============================================================
# 7. RUN APIFY ACTOR
# ============================================================

print(
    "\nScraping Instagram profile..."
)

try:

    run = client.actor(
        "apify/instagram-profile-scraper"
    ).call(
        run_input=run_input
    )

except Exception as e:

    print(
        "\nApify scraping failed:"
    )

    print(e)

    raise SystemExit


# ============================================================
# 8. GET DATASET ID
# ============================================================

if isinstance(
    run,
    dict
):

    dataset_id = run.get(
        "defaultDatasetId"
    )

else:

    dataset_id = getattr(
        run,
        "default_dataset_id",
        None
    )


if not dataset_id:

    print(
        "\nCould not find Apify dataset ID."
    )

    print(
        "Run information:"
    )

    print(run)

    raise SystemExit


# ============================================================
# 9. GET DATASET
# ============================================================

try:

    dataset = client.dataset(
        dataset_id
    )

    items = dataset.list_items().items

except Exception as e:

    print(
        "\nCould not retrieve Apify dataset:"
    )

    print(e)

    raise SystemExit


# ============================================================
# 10. PROFILE NOT FOUND
# ============================================================

if not items:

    print(
        "\n"
        + "=" * 60
    )

    print(
        "PROFILE DATA NOT AVAILABLE"
    )

    print(
        "=" * 60
    )

    print(
        f"\nUsername: {username}"
    )

    print(
        "\nApify could not retrieve this Instagram profile."
    )

    print(
        "This does NOT mean the account is fake."
    )

    print(
        "\nPrediction cancelled."
    )

    raise SystemExit


# ============================================================
# 11. PROFILE DATA
# ============================================================

data = items[0]


print(
    "\n"
    + "=" * 60
)

print(
    "PROFILE INFORMATION"
)

print(
    "=" * 60
)

print(
    "Username:",
    data.get("username")
)

print(
    "Full Name:",
    data.get("fullName")
)

print(
    "Followers:",
    data.get("followersCount")
)

print(
    "Following:",
    data.get("followsCount")
)

print(
    "Posts:",
    data.get("postsCount")
)

print(
    "Biography:",
    data.get("biography")
)

print(
    "External URL:",
    data.get("externalUrl")
)

print(
    "Profile Picture:",
    bool(
        data.get(
            "profilePicUrl"
        )
    )
)


# ============================================================
# 12. POSTS
# ============================================================

posts = data.get(
    "latestPosts",
    []
)


print(
    "\n"
    + "=" * 60
)

print(
    "POST INFORMATION"
)

print(
    "=" * 60
)

print(
    "Number of posts received:",
    len(posts)
)


# ============================================================
# 13. CHECK POST DATA AVAILABILITY
# ============================================================

actual_post_count = (
    data.get(
        "postsCount",
        0
    )
    or 0
)


# ------------------------------------------------------------
# CASE 1:
# Account has posts, but Apify didn't return them
# ------------------------------------------------------------

if (
    actual_post_count > 0
    and len(posts) == 0
):

    print(
        "\n"
        + "=" * 60
    )

    print(
        "POST DATA UNAVAILABLE"
    )

    print(
        "=" * 60
    )

    print(
        f"\nInstagram reports {actual_post_count} posts."
    )

    print(
        "However, Apify returned 0 latest posts."
    )

    print(
        "\nWe CANNOT assume:"
    )

    print(
        "Likes = 0"
    )

    print(
        "Comments = 0"
    )

    print(
        "Engagement = 0"
    )

    print(
        "Hashtags = 0"
    )

    print(
        "Locations = 0"
    )

    print(
        "Post interval = 0"
    )

    print(
        "\nTherefore the current 15-feature model"
    )

    print(
        "will NOT be used for this account."
    )

    print(
        "\nPrediction: INSUFFICIENT DATA"
    )

    print(
        "Reason: Instagram posts could not be retrieved."
    )

    raise SystemExit


# ------------------------------------------------------------
# CASE 2:
# Account genuinely has zero posts
# ------------------------------------------------------------

if (
    actual_post_count == 0
):

    print(
        "\n"
        + "=" * 60
    )

    print(
        "ZERO-POST ACCOUNT"
    )

    print(
        "=" * 60
    )

    print(
        "\nThis account genuinely has 0 posts."
    )

    print(
        "The current 15-feature model was not trained"
    )

    print(
        "to safely classify zero-post accounts."
    )

    print(
        "\nPrediction: INSUFFICIENT DATA"
    )

    print(
        "Reason: Current model requires post-derived features."
    )

    raise SystemExit


# ============================================================
# 14. PRINT POST DETAILS
# ============================================================

print(
    "\n========== POST TYPES =========="
)

for i, post in enumerate(
    posts,
    start=1
):

    print(
        f"Post {i}: "
        f"Type={post.get('type')} | "
        f"Likes={post.get('likesCount')} | "
        f"Comments={post.get('commentsCount')} | "
        f"Timestamp={post.get('timestamp')}"
    )


# ============================================================
# 15. FEATURE ENGINEERING
# ============================================================

features = calculate_features(
    data
)


print(
    "\n"
    + "=" * 60
)

print(
    "15 ML FEATURES"
)

print(
    "=" * 60
)

print(
    features.to_string(
        index=False
    )
)


# ============================================================
# 16. CHECK FOR MISSING FEATURES
# ============================================================

if features[
    FEATURE_NAMES
].isnull().any().any():

    print(
        "\n"
        + "=" * 60
    )

    print(
        "MISSING FEATURE DATA"
    )

    print(
        "=" * 60
    )

    print(
        "\nSome post-derived features are unavailable."
    )

    print(
        "Prediction cancelled."
    )

    raise SystemExit


# ============================================================
# 17. CHECK FEATURE ORDER
# ============================================================

if list(
    features.columns
) != FEATURE_NAMES:

    print(
        "\nFeature order mismatch."
    )

    print(
        "Expected:"
    )

    print(
        FEATURE_NAMES
    )

    print(
        "Received:"
    )

    print(
        list(features.columns)
    )

    raise SystemExit


print(
    "\n========== FEATURE ORDER =========="
)

print(
    list(
        features.columns
    )
)


# ============================================================
# 18. SCALE FEATURES
# ============================================================

try:

    features_scaled = scaler.transform(
        features[
            FEATURE_NAMES
        ]
    )

except Exception as e:

    print(
        "\nFeature scaling failed:"
    )

    print(e)

    raise SystemExit


# ============================================================
# 19. KEEP FEATURE NAMES
# ============================================================

features_scaled_df = pd.DataFrame(
    features_scaled,
    columns=FEATURE_NAMES
)


# ============================================================
# 20. MODEL PREDICTION
# ============================================================

try:

    prediction = model.predict(
        features_scaled_df
    )

    probabilities = model.predict_proba(
        features_scaled_df
    )

except Exception as e:

    print(
        "\nModel prediction failed:"
    )

    print(e)

    raise SystemExit


# ============================================================
# 21. PROBABILITIES
# ============================================================

real_probability = probabilities[0][0]

fake_probability = probabilities[0][1]


# ============================================================
# 22. FINAL PREDICTION
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "PREDICTION RESULT"
)

print(
    "=" * 60
)


if prediction[0] == 1:

    print(
        "\nPrediction: Fake Instagram Account"
    )

    print(
        f"Confidence: "
        f"{fake_probability * 100:.2f}%"
    )

else:

    print(
        "\nPrediction: Real Instagram Account"
    )

    print(
        f"Confidence: "
        f"{real_probability * 100:.2f}%"
    )


# ============================================================
# 23. PROBABILITY BREAKDOWN
# ============================================================

print(
    "\n========== PROBABILITIES =========="
)

print(
    f"Real Probability: "
    f"{real_probability * 100:.2f}%"
)

print(
    f"Fake Probability: "
    f"{fake_probability * 100:.2f}%"
)


# ============================================================
# 24. FEATURE IMPORTANCE
# ============================================================

if hasattr(
    model,
    "feature_importances_"
):

    print(
        "\n========== MODEL FEATURE IMPORTANCE =========="
    )

    importance_pairs = sorted(
        zip(
            FEATURE_NAMES,
            model.feature_importances_
        ),
        key=lambda x: x[1],
        reverse=True
    )

    for feature, importance in importance_pairs:

        print(
            f"{feature:5} : "
            f"{importance:.6f}"
        )


print(
    "\nAnalysis completed."
)