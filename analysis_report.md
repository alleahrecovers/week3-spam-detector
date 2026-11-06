1. Features Selected

My spam detector uses six features: word count, exclamation marks, money-related words, all-caps words, URLs, and urgency words.

I added two custom features: Has URL, which checks for website links, and Urgency Words, which looks for words such as "urgent," "immediately," and "act fast." I chose these because spam emails often use suspicious links and pressure tactics.

2. Model Results

I trained a Logistic Regression model using 16 emails: 8 spam and 8 legitimate. I used 75% for training and 25% for testing.

Accuracy: 100%

Precision: 100%

Recall: 100%


The model correctly classified all four test emails. However, the test dataset was very small, so these results may not represent real-world performance.

3. Most Important Error

I believe false positives are more concerning because they cause legitimate emails to be marked as spam. This could result in someone missing important messages from school, work, or other contacts.

4. Feature Importance

Exclamation marks had the strongest influence on the model, followed by money-related words and all-caps words. URLs had no measurable influence in this model.

5. Tricky Emails and Improvements

Of the three tricky emails, the legitimate project report was incorrectly classified as spam. The other two were classified correctly.

In my variation test, adding three exclamation marks to a legitimate meeting email changed its prediction from legitimate to spam. This shows that the model is sensitive to punctuation.
