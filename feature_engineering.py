import numpy as np
import pandas as pd


PROMOTIONAL_KEYWORDS = {
    "regrann",
    "contest",
    "repost",
    "giveaway",
    "mention",
    "share",
    "give away",
    "quiz"
}


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


def get_caption(post):

    caption = post.get("caption")

    if caption is None:
        return ""

    return str(caption)


def get_hashtags(post):

    hashtags = post.get("hashtags", [])

    if not hashtags:
        return []

    if isinstance(hashtags, list):

        return [
            str(tag).lower().lstrip("#")
            for tag in hashtags
        ]

    return []


def is_location_available(post):

    location = post.get("locationName")

    if location:
        return True

    location = post.get("location")

    if location:
        return True

    location = post.get("locationId")

    if location:
        return True

    return False


def calculate_features(profile):

    # =====================================================
    # PROFILE FEATURES
    # =====================================================

    pos = profile.get(
        "postsCount",
        0
    ) or 0

    flw = profile.get(
        "followersCount",
        0
    ) or 0

    flg = profile.get(
        "followsCount",
        0
    ) or 0

    biography = profile.get(
        "biography",
        ""
    ) or ""

    bl = len(
        str(biography)
    )

    pic = 1 if profile.get(
        "profilePicUrl"
    ) else 0

    lin = 1 if profile.get(
        "externalUrl"
    ) else 0


    # =====================================================
    # LATEST POSTS
    # =====================================================

    posts = profile.get(
        "latestPosts",
        []
    )

    # -----------------------------------------------------
    # IMPORTANT:
    #
    # We DO NOT assume that missing posts mean zero
    # engagement.
    #
    # If posts are unavailable, these values are only
    # placeholders and test_apify.py will NOT send them
    # to the model.
    # -----------------------------------------------------

    if not posts:

        features = pd.DataFrame([{

            "pos": pos,
            "flw": flw,
            "flg": flg,
            "bl": bl,
            "pic": pic,
            "lin": lin,

            "cl": np.nan,
            "cz": np.nan,
            "ni": np.nan,
            "erl": np.nan,
            "erc": np.nan,
            "lt": np.nan,
            "hc": np.nan,
            "pr": np.nan,
            "pi": np.nan

        }])

        return features


    # =====================================================
    # CAPTION FEATURES
    # =====================================================

    captions = [
        get_caption(post)
        for post in posts
    ]

    caption_lengths = [
        len(caption)
        for caption in captions
    ]

    cl = np.mean(
        caption_lengths
    )


    # =====================================================
    # SHORT CAPTION RATIO
    # =====================================================

    short_caption_count = sum(
        1
        for length in caption_lengths
        if length <= 3
    )

    cz = (
        short_caption_count
        / len(posts)
    )


    # =====================================================
    # NON-IMAGE / MEDIA TYPE RATIO
    # =====================================================

    non_image_count = 0

    for post in posts:

        post_type = str(
            post.get(
                "type",
                ""
            )
        ).lower()

        if post_type in [
            "video",
            "sidecar",
            "carousel"
        ]:

            non_image_count += 1

    ni = (
        non_image_count
        / len(posts)
    )


    # =====================================================
    # LIKE ENGAGEMENT RATE
    # =====================================================

    total_likes = sum(
        (
            post.get(
                "likesCount",
                0
            ) or 0
        )
        for post in posts
    )

    if flw > 0:

        erl = (
            total_likes
            /
            (
                len(posts)
                * flw
            )
        )*100

    else:

        erl = 0


    # =====================================================
    # COMMENT ENGAGEMENT RATE
    # =====================================================

    total_comments = sum(
        (
            post.get(
                "commentsCount",
                0
            ) or 0
        )
        for post in posts
    )

    if flw > 0:

        erc = (
            total_comments
            /
            (
                len(posts)
                * flw
            )
        )*100

    else:

        erc = 0


    # =====================================================
    # LOCATION RATIO
    # =====================================================

    location_count = sum(
        1
        for post in posts
        if is_location_available(post)
    )

    lt = (
        location_count
        / len(posts)
    )


    # =====================================================
    # HASHTAG COUNT
    # =====================================================

    hashtag_counts = [
        len(
            get_hashtags(post)
        )
        for post in posts
    ]

    hc = np.mean(
        hashtag_counts
    )


    # =====================================================
    # PROMOTIONAL KEYWORDS
    # =====================================================

    promotional_usage = []

    for post in posts:

        hashtags = get_hashtags(
            post
        )

        count = 0

        for hashtag in hashtags:

            hashtag_clean = (
                hashtag
                .lower()
                .strip()
            )

            if (
                hashtag_clean
                in PROMOTIONAL_KEYWORDS
            ):

                count += 1

        promotional_usage.append(
            count
        )

    pr = np.mean(
        promotional_usage
    )


    # =====================================================
    # POST INTERVAL
    # =====================================================

    timestamps = []

    for post in posts:

        timestamp = post.get(
            "timestamp"
        )

        if timestamp:

            try:

                dt = pd.to_datetime(
                    timestamp
                )

                timestamps.append(
                    dt
                )

            except Exception:

                pass


    timestamps.sort()

    intervals = []

    for i in range(
        1,
        len(timestamps)
    ):

        difference = (
            timestamps[i]
            - timestamps[i - 1]
        ).total_seconds() / 3600

        intervals.append(
            abs(difference)
        )


    if intervals:

        pi = np.mean(
            intervals
        )

    else:

        pi = 0


    # =====================================================
    # FINAL 15 FEATURES
    # =====================================================

    features = pd.DataFrame([{

        "pos": pos,

        "flw": flw,

        "flg": flg,

        "bl": bl,

        "pic": pic,

        "lin": lin,

        "cl": cl,

        "cz": cz,

        "ni": ni,

        "erl": erl,

        "erc": erc,

        "lt": lt,

        "hc": hc,

        "pr": pr,

        "pi": pi

    }])

    return features