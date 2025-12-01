from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

sentences = [
    # Bug reports
    "The app crashes when I click the login button",
    "There is an error when uploading files",
    "The page does not load properly",
    "Getting 404 error on the main page",
    "Database connection timeout issue",
    "Form validation is not working correctly",
    "Memory leak causing slow performance",
    "Images are not displaying in the gallery",
    "Search function returns incorrect results",
    "User authentication fails randomly",
    "Mobile app freezes during data sync",
    "Payment processing throws an exception",
    "The submit button is not responding",
    "Getting blank screen after login",
    "File download is corrupted",
    "Session expires too quickly",
    "Navigation menu is broken on mobile",
    "Email notifications are not being sent",
    "Data is not saving properly",
    "Calendar events are displaying wrong dates",
    "Profile picture upload fails silently",
    "Password reset link is not working",
    "Charts are not rendering correctly",
    "API calls are returning 500 errors",
    "Text is overlapping in the sidebar",
    "Video playback stutters and lags",
    "Drag and drop functionality is broken",
    "Auto-save feature is not working",
    "Print preview shows garbled text",
    "Infinite loading spinner on dashboard",

    # Feature requests
    "Please add dark mode support",
    "Can we have a new dashboard layout?",
    "Add option to export reports",
    "Need integration with third-party APIs",
    "Implement real-time notifications",
    "Add multi-language support",
    "Create user role management system",
    "Include data visualization charts",
    "Add bulk operations for data management",
    "Implement advanced search filters",
    "Need offline mode functionality",
    "Add social media login options",
    "Please add a calendar integration feature",
    "Can we get email notification settings?",
    "Add drag and drop file upload",
    "Implement two-factor authentication",
    "Need a mobile app version",
    "Add custom themes and branding",
    "Include automated backup functionality",
    "Add collaborative editing features",
    "Implement voice command support",
    "Need barcode scanning capability",
    "Add GPS location tracking",
    "Include video call integration",
    "Add inventory management module",
    "Implement AI-powered recommendations",
    "Need advanced reporting dashboard",
    "Add QR code generation feature",
    "Include document version control",
    "Add automated workflow triggers",

    # Other/General feedback
    "The interface looks clean",
    "Great work on the latest update",
    "I like the new color scheme",
    "Documentation is very helpful",
    "The onboarding process is smooth",
    "Performance has improved significantly",
    "User experience is intuitive",
    "The design is modern and professional",
    "Loading times are much faster now",
    "Customer support is responsive",
    "The tutorial videos are clear",
    "Overall satisfaction with the product",
    "Thank you for the quick response",
    "The team is doing excellent work",
    "Really appreciate the help",
    "This tool has been very useful",
    "Love the attention to detail",
    "The app feels very polished",
    "Impressed with the quality",
    "Keep up the good work",
    "The updates are always welcome",
    "Very satisfied with the service",
    "The platform is reliable",
    "Great job on the improvements",
    "The workflow is very efficient",
    "Excellent customer experience",
    "The interface is user-friendly",
    "Really enjoying using this app",
    "The features work as expected",
    "Very happy with the results"
]

labels = [
    # Bug reports (30 samples)
    "bug", "bug", "bug", "bug", "bug", "bug",
    "bug", "bug", "bug", "bug", "bug", "bug",
    "bug", "bug", "bug", "bug", "bug", "bug",
    "bug", "bug", "bug", "bug", "bug", "bug",
    "bug", "bug", "bug", "bug", "bug", "bug",

    # Feature requests (30 samples)
    "feature", "feature", "feature", "feature", "feature", "feature",
    "feature", "feature", "feature", "feature", "feature", "feature",
    "feature", "feature", "feature", "feature", "feature", "feature",
    "feature", "feature", "feature", "feature", "feature", "feature",
    "feature", "feature", "feature", "feature", "feature", "feature",

    # Other/General feedback (30 samples)
    "other", "other", "other", "other", "other", "other",
    "other", "other", "other", "other", "other", "other",
    "other", "other", "other", "other", "other", "other",
    "other", "other", "other", "other", "other", "other",
    "other", "other", "other", "other", "other", "other"
]

X_train, X_test, y_train, y_test = train_test_split(
    sentences, labels, test_size=0.2, random_state=42
)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)
print("Predictions:", predictions)

test_examples = [
    "The app keeps freezing",
    "Please add a filter option",
    'please add a nice logo to the app',
    'the logo should be bigger and clearer',
    'the logo should be smaller and clearer',
    'the logo should be bigger and clearer and should be in the center of the screen',
    'the logo should be smaller and clearer and should be in the center of the screen',
    'the logo should be bigger and clearer and should be in the center of the screen',
    'the logo should be smaller and clearer and should be in the center of the screen',
    "Nice work team",
    "Add push notifications for new messages",
    "Need a shopping cart feature",
    "Can you implement user profiles?",
    "Add photo gallery functionality",
    "Include weather widget integration",
    "Need password reset functionality",
    "Add file compression before upload",
    "Implement chat bot support",
    "Need print functionality for reports",
    "Add bookmark saving feature"
]

results = model.predict(test_examples)

print("\nNew Predictions:")
for text, label in zip(test_examples, results):
    print(f"Sentence: {text}  ->  Predicted: {label}")
