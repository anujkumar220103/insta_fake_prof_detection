Fake Instagram Account Detection - Dataset Explanation
#  Instagram Fake Account Detection Dataset – README

## Dataset Overview

* **Total Columns:** 17
  → 16 Input Features + 1 Target Variable

* **Removed Feature:** `fo (Followers Keywords)`

  * Reason: Low contribution and negligible impact on model performance

* **Target Variable:** `class`

  * `0 = Real Account`
  * `1 = Fake Account`

---

##  Feature Description

### 1. pos (Posts Count)

* Total number of posts made by the account
* **Real:** Usually has multiple posts
* **Fake:** Often 0 or very few posts
* **Importance:** High ✅

---

### 2. flw (Followers Count)

* Number of followers
* **Real:** Grows naturally over time
* **Fake:** Very low or artificially high
* **Importance:** Medium

---

### 3. flg (Following Count)

* Number of accounts followed
* **Fake:** Often follows many users (mass following)
* **Importance:** High ✅

---

### 4. bl (Biography Length)

* Number of characters in bio
* **Real:** Contains meaningful information
* **Fake:** Empty or random text
* **Importance:** Low–Medium

---

### 5. pic (Profile Picture Availability)

* `1 = Profile picture present`
* `0 = No profile picture`
* **Real:** Usually present
* **Fake:** Often missing
* **Importance:** Low

---

### 6. lin (External Link in Bio)

* `1 = Link present`
* `0 = No link`
* **Real:** Sometimes present
* **Fake:** May include spam links
* **Importance:** Low

---

### 7. cl (Average Caption Length)

* Average number of characters in captions
* **Real:** Meaningful captions
* **Fake:** Very short or repetitive
* **Importance:** Medium

---

### 8. cz (Caption Zero Percentage)

* Percentage of captions with very low length (≤ 3 characters)
* **Real:** Low percentage
* **Fake:** High percentage
* **Importance:** Medium

---

### 9. ni (Non-Image Content Percentage)

* Percentage of posts that are not images (videos/carousels)
* Helps analyze content type distribution
* **Importance:** Medium

---

### 10. erl (Engagement Rate – Likes)

* Formula:
  ER(likes) = Total Likes / (Posts × Followers)
* **Real:** Balanced engagement
* **Fake:** Very low engagement
* **Importance:** High ✅

---

### 11. erc (Engagement Rate – Comments)

*  ER(comments) = Total commnets / (Posts × Followers)
* **Real:** Meaningful interaction
* **Fake:** Very low or no comments
* **Importance:** High ✅

---

### 12. lt (Location Tag Percentage)

* Percentage of posts with location tags
* **Real:** Often includes locations
* **Fake:** Rarely uses location tags
* **Importance:** Medium

---

### 13. hc (Average Hashtag Count)

* Average number of hashtags per post
* **Fake:** Often uses excessive hashtags
* **Importance:** Medium

---

### 14. pr (Promotional Keywords Usage)

* Frequency of promotional keywords (e.g., giveaway, repost, contest)
* **Fake:** High usage indicates spam behavior
* **Importance:** Medium

---

### 15. cs (Content Similarity)

* Measures similarity between posts (cosine similarity)
* **Fake:** Repetitive content
* **Real:** Diverse content
* **Importance:** High ✅

---

### 16. pi (Post Interval)

* Average time gap between posts (in hours)
* **Fake:** Irregular or too frequent posting
* **Importance:** High ✅

---

### 17. class (Target Variable)

* Output label for classification

  * `0 → Real Account`
  * `1 → Fake Account`

---






# Feature Importance Summary (Presentation)

## Highly Important Features

• flg (Following Count)
• pos (Posts Count)
• erl (Engagement Rate – Likes)
• erc (Engagement Rate – Comments)
• cs (Content Similarity)
• pi (Post Interval)

---
## Medium Important Features

• flw (Followers Count)
• cl (Caption Length)
• lt (Location Tag Percentage)
• hc (Average Hashtag Count)
• cz (Caption Zero Percentage)
• ni (Non-Image Percentage)

---

##  Least Important Features
 
• pic (Profile Picture)
• bl (Bio Length)
• lin (External Link in Bio)
• pr (Promotional Keywords)

---

##  Feature Count Summary

* Dependent Feature (Target): 1 → class
* Independent Features (Input): 16






references:- 

paper - 

[1] P. K. Roy and S. Chahar, 
"Fake Profile Detection on Social Networking Websites: A Comprehensive Review," 
IEEE Transactions on Artificial Intelligence, vol. 1, no. 3, pp. 271–281, 2020.

[2] K. R. Purba, D. Asirvatham, and R. K. Murugesan, 
"Classification of Instagram fake users using supervised machine learning algorithms," 
International Journal of Electrical and Computer Engineering, vol. 10, no. 3, pp. 2763–2772, 2020.

dataset - 

[3] K. R. Purba, "Fake and authentic Instagram users dataset," 
Kaggle, 2020. [Online]. Available: https://www.kaggle.com/datasets/krpurba/fakeauthentic-user-instagram






